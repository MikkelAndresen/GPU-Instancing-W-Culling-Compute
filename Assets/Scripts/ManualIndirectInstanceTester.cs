using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Rendering;
using System;
using Unity.Jobs;
using Unity.Profiling;
using CommonJobs;

// Sort Input matrix buffer by a priority based on how often the developer thinks they will be updated.
// e.g if 10 objects will update often, put them in index 0 - 10.
// Cache all updates for any given frame in a index list, loop to find the min/max indexes to be updated,
// then create a subset to update the buffer.
// This could also we done by a job.

public class ManualIndirectInstanceTester : MonoBehaviour
{
	// According to this source (https://forum.unity.com/threads/clarification-on-computebuffermode-immutable.1150250/),
	// the dynamic compute buffer mode is only for meshes and not for custom data which we use. From my own testing,
	// I wasn't able to make dynamic work with my data. Which is why I made a custom enum for serialized setup of the test script.
	public enum ComputeBufferWriteMode
	{
		Immutable,
		SubUpdates
	}

	public enum GPUWriteMethod
	{
		NativeArrayCopy,
		ParallelForJobSet,
		JobNativeArrayCopy,
		ParallelBatchJobCopy
	}

	[SerializeField]
	private Mesh mesh;
	[SerializeField]
	private int dimension = 10;
	[SerializeField]
	private int serializedTotalCount;
	[SerializeField]
	private float space = 1.5f;

	[SerializeField]
	private int jobBatchDiv = 10;
	[SerializeField, SetProperty(nameof(GPUCopyMethod))]
	private GPUWriteMethod gpuCopyMethod = GPUWriteMethod.ParallelBatchJobCopy;
	public GPUWriteMethod GPUCopyMethod
	{
		get => gpuCopyMethod;
		set
		{
			if (!Application.isPlaying)
				return;

			FinishGPU_UploadJob();
			gpuCopyMethod = value;
		}
	}

	[SerializeField, SetProperty(nameof(MatrixBufferWriteMethod))]
	private ComputeBufferWriteMode matrixBufferWriteMethod = ComputeBufferWriteMode.Immutable;
	public ComputeBufferWriteMode MatrixBufferWriteMethod
	{
		get => matrixBufferWriteMethod;
		set
		{
			if (!Application.isPlaying)
				return;

			FinishGPU_UploadJob();
			InvalidateMatrixBuffer();
			matrixBufferWriteMethod = value;
		}
	}

	[SerializeField]
	private ShadowCastingMode shadowCastingMode = ShadowCastingMode.On;
	[SerializeField]
	private bool receiveShadows = true;
	[SerializeField, SetProperty(nameof(UploadDataStartOfNextFrame)),
	Tooltip("If checked will make sure jobs only complete at the beginning of next frame, this can improve performance but increases latency")]
	private bool uploadDataStartOfNextFrame = false;
	public bool UploadDataStartOfNextFrame
	{
		get => uploadDataStartOfNextFrame;
		set
		{
			if (!Application.isPlaying)
				return;

			FinishGPU_UploadJob();
			uploadDataStartOfNextFrame = value;
		}
	}

	[SerializeField]
	private Shader instancingShader;
	private Shader InstancingShader => instancingShader == null ? Shader.Find("Unlit/InstancedIndirectUnlit") : instancingShader;

	private ComputeBufferMode MatrixBufferMode =>
		matrixBufferWriteMethod == ComputeBufferWriteMode.Immutable ? ComputeBufferMode.Immutable : ComputeBufferMode.SubUpdates;
	public int TotalCount => dimension * dimension * dimension;
	public bool UseSubUpdates => matrixBufferWriteMethod == ComputeBufferWriteMode.SubUpdates;

	private Bounds bounds = new Bounds(Vector3.zero, Vector3.one * 10000);
	private uint[] args;
	private ComputeBuffer argsBuffer;
	private Material mat;

	private GenericNativeComputeBuffer<float3x4> matrixInputBuffer;
	private GenericNativeComputeBuffer<uint> indexOutputBuffer;
	private NativeArray<int> parallelCounter; // Used to hold a single counter for indexing
	private GenericNativeComputeBuffer<float4> colorBuffer;

	private TestDataGenerator dataGen;
	private DataSubset matrixBufferSubset = new DataSubset(0, 0);
	private DataSubset colorBufferSubset = new DataSubset(0, 0);
	private ComputeCuller computeCuller;
	private JobHandle currentGPUCopyJob;

	protected static int materialMatrixBufferID = Shader.PropertyToID("matrixBuffer");
	protected static int materialIndexBufferID = Shader.PropertyToID("indexBuffer");
	protected static int colorBufferID = Shader.PropertyToID("colorBuffer");

	void Start()
	{
		serializedTotalCount = TotalCount;

		if (mat == null)
		{
			mat = new Material(InstancingShader)
			{
				//This alone doesn't actually work to set required keywords when using Graphics.DrawMeshInstancedIndirect
				// Instancing keywords are required for the CustomLit shader as I use the builtin unity_InstanceID field.
				// Those required keywords are somehow provided when I add a setup function to each shader pass, see the CustomLit.shader file for details.
				enableInstancing = true
			};
		}

		indexOutputBuffer = new GenericNativeComputeBuffer<uint>(
			new NativeArray<uint>(TotalCount, Allocator.Persistent), ComputeBufferType.Append, ComputeBufferMode.Immutable);

		parallelCounter = new NativeArray<int>(1, Allocator.Persistent);

		dataGen = new TestDataGenerator(dimension);

		if (Camera.main != null)
			computeCuller = new ComputeCuller(Camera.main);
		else
			Debug.LogError($"{nameof(ManualIndirectInstanceTester)} failed to create compute culler because there is no main camera");

		colorBuffer = new GenericNativeComputeBuffer<float4>(dataGen.colors);

		InvalidateMatrixBuffer();

		// Set all inital data
		dataGen.RunMatrixJob(dimension, space, true, Time.deltaTime);
		dataGen.RunColorJob(dimension, true);
		SetMatrixBufferData();
		SetColorBufferData();

		mat.SetBuffer(materialIndexBufferID, indexOutputBuffer.Buffer);
		mat.SetBuffer(colorBufferID, colorBuffer.Buffer);

		// Set initial argument buffer for DrawMeshInstancedIndirect
		InvalidateArgumentBuffer();
	}

	private void InvalidateMatrixBuffer()
	{
		if (matrixInputBuffer != null)
			matrixInputBuffer.Dispose();

		matrixInputBuffer = new GenericNativeComputeBuffer<float3x4>(
			new NativeArray<float3x4>(TotalCount, Allocator.Persistent), ComputeBufferType.Default, MatrixBufferMode);
		SetMatrixBufferData();
		mat.SetBuffer(materialMatrixBufferID, matrixInputBuffer.Buffer);
	}

	private readonly ProfilerMarker completeDataGenerationJobMarker = new ProfilerMarker("Complete Data generation");
	void Update()
	{
		// This is safe no matter what our condition as we always want this finished by this point
		// and there is no harm if it's already finished.
		// It also means that forced updates from the editor (which can happen when interacting with the inspector),
		// can't cause issues with arrays used in this job and the testData generate job.
		CompleteGPUCopyJob();

		ResetIndexBufferCounter();

		if (uploadDataStartOfNextFrame)
			UploadGPUDataAndDispatchCompute();

		// We isolate the data generation so we clearly see other costs
		completeDataGenerationJobMarker.Begin();
		dataGen.RunMatrixJob(dimension, space, completeNow: true, Time.deltaTime);
		completeDataGenerationJobMarker.End();

		// Performance tests for pushing data to GPU
		if (TotalCount > 0)
			StartWriteJob();
	}

	private void LateUpdate()
	{
		if (!uploadDataStartOfNextFrame)
			UploadGPUDataAndDispatchCompute();

		Render();
	}

	private void InvalidateArgumentBuffer()
	{
		args = new uint[]
		{
			mesh.GetIndexCount(0),
			(uint)(TotalCount),
			mesh.GetIndexStart(0),
			mesh.GetBaseVertex(0),
			0
		};
		argsBuffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
		argsBuffer.SetData(args);
	}

	private void ResetIndexBufferCounter() => indexOutputBuffer.Buffer.SetCounterValue(0);

	private void UpdateAppendCountInArgs()
	{
		ComputeBuffer.CopyCount(indexOutputBuffer.Buffer, argsBuffer, 4);
	}

	private readonly ProfilerMarker renderMarker = new ProfilerMarker("Do Render");
	public void Render()
	{
		renderMarker.Begin();

		Graphics.DrawMeshInstancedIndirect(
				mesh,
				submeshIndex: 0,
				mat,
				bounds,
				argsBuffer,
				argsOffset: 0,
				properties: null,
				shadowCastingMode,
				receiveShadows: receiveShadows,
				layer: gameObject.layer,
				camera: null,
				LightProbeUsage.BlendProbes,
				lightProbeProxyVolume: null);

		renderMarker.End();
	}

	private readonly ProfilerMarker beginGPUWriteMarker = new ProfilerMarker("BeginGPU Write");
	private Action endBufferWrite;
	private void StartWriteJob()
	{
		if (UseSubUpdates)
		{
			beginGPUWriteMarker.Begin();
			endBufferWrite = matrixInputBuffer.BeginWriteMatrices<float3x4>(0, TotalCount, CopyDataToGPU);
			beginGPUWriteMarker.End();
		}
		else
		{
			// Copy test data to the gpu array
			var array = matrixInputBuffer.Data;
			CopyDataToGPU(ref array);
		}
	}
	private NativeArray<Plane> nativeFrustumPlanes;

	private readonly ProfilerMarker copyToGPUMarker = new ProfilerMarker("Copy data to GPU");
	private void CopyDataToGPU(ref NativeArray<float3x4> array)
	{
		copyToGPUMarker.Begin();

		switch (gpuCopyMethod)
		{
			case GPUWriteMethod.NativeArrayCopy:
				array.CopyFrom(dataGen.matrices);
				break;
			case GPUWriteMethod.ParallelForJobSet:
				int batchCount = TotalCount / 4;
				currentGPUCopyJob = new ParallelCPUToGPUCopyJob<float3x4>()
				{
					src = dataGen.matrices,
					dst = array
				}.Schedule(TotalCount, batchCount);
				break;
			case GPUWriteMethod.JobNativeArrayCopy:
				currentGPUCopyJob = new CPUToGPUCopyJob<float3x4>()
				{
					src = dataGen.matrices,
					dst = array
				}.Schedule();
				break;
			case GPUWriteMethod.ParallelBatchJobCopy:
				int length = TotalCount;
				currentGPUCopyJob = new BatchParallelCPUToGPUCopyJob<float3x4>()
				{
					src = dataGen.matrices,
					dst = array
				}.ScheduleBatch(length, length / jobBatchDiv);
				break;
			default:
				break;
		}

		copyToGPUMarker.End();
	}

	private readonly ProfilerMarker completeCopyJobMarker = new ProfilerMarker("Completed GPU Copy Job");
	private void CompleteGPUCopyJob()
	{
		completeCopyJobMarker.Begin();
		// Finish gpu copy job
		currentGPUCopyJob.Complete();
		completeCopyJobMarker.End();
	}

	private readonly ProfilerMarker endGPUWriteMarker = new ProfilerMarker("Finished writing GPU data");
	private void FinishGPU_UploadJob()
	{
		CompleteGPUCopyJob();

		endGPUWriteMarker.Begin();
		// Finish ComputeBuffer.BeginWrite
		endBufferWrite?.Invoke();
		// Set to null so there is no chance of reuse, otherwise we get an exception about the array being deallocated
		endBufferWrite = null;
		endGPUWriteMarker.End();
	}

	private void UploadGPUDataAndDispatchCompute()
	{
		FinishGPU_UploadJob();

		// Dispatch culling compute after we've uploaded the matrices
		computeCuller.DispatchCompute(matrixInputBuffer.Buffer, indexOutputBuffer.Buffer);
		if (!UseSubUpdates)
			SetBufferData();

		UpdateAppendCountInArgs();
	}

	private void SetBufferData()
	{
		// ComputeBuffer SetData
		if (!UseSubUpdates)
			SetMatrixBufferData();
	}

	private readonly ProfilerMarker pushAllMatricesMarker = new ProfilerMarker("ComputeBuffer.SetData");
	private void SetMatrixBufferData()
	{
		pushAllMatricesMarker.Begin();
		if (matrixBufferSubset.count != TotalCount - 1)
			matrixBufferSubset = new DataSubset(0, TotalCount - 1);

		if (matrixInputBuffer.SetData(matrixBufferSubset))
			mat.SetBuffer(materialMatrixBufferID, matrixInputBuffer.Buffer);

		pushAllMatricesMarker.End();
	}

	private readonly ProfilerMarker pushAllColorsMarker = new ProfilerMarker("Update Colors and Set ComputeBuffer");
	private void SetColorBufferData()
	{
		pushAllColorsMarker.Begin();
		if (colorBufferSubset.count != TotalCount - 1)
			colorBufferSubset = new DataSubset(0, TotalCount - 1);
		colorBuffer.SetData(colorBufferSubset);
		pushAllColorsMarker.Begin();
	}

	private void OnDestroy()
	{
		// We wanna finish the jobs here in case it's still running
		// because they depend on stuff we dispose below
		FinishGPU_UploadJob();

		matrixInputBuffer.Dispose();
		colorBuffer.Dispose();
		argsBuffer.Release();
		indexOutputBuffer.Dispose();
		dataGen.Dispose();
		parallelCounter.Dispose();
		computeCuller.Dispose();
	}
}