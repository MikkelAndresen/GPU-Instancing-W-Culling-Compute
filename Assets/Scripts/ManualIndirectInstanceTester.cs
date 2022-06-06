using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
using System;
using Unity.Jobs;
using Unity.Profiling;
using System.Threading;
using Unity.Burst;
using System.Collections;

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
	private bool updateAllMatrices, updateSingleMatrix, updateMatrixSubset;
	[SerializeField]
	private int jobInnerBatchCount = 1;
	[SerializeField]
	private GPUWriteMethod gpuWriteMethod = GPUWriteMethod.ParallelBatchJobCopy;
	[SerializeField]
	private ComputeBufferWriteMode matrixBufferMode = ComputeBufferWriteMode.Immutable;
	[SerializeField]
	private ShadowCastingMode shadowCastingMode = ShadowCastingMode.On;
	[SerializeField]
	private bool receiveShadows = true;
	[SerializeField, Tooltip("If checked will ensure gpu data is only uploaded at EndOfFrame rather than late update, this means that data from the current frame will only be visible next frame")]
	private bool uploadGPUDataEndOfFrame = false;
	[SerializeField]
	private int dispatchX;
	[SerializeField]
	private float jobDeltaTime;
	private float jobStartTimeSeconds;
	[SerializeField]
	private Shader instancingShader;
	private Shader InstancingShader => instancingShader == null ? Shader.Find("Unlit/InstancedIndirectUnlit") : instancingShader;

	private ComputeBuffer InputBuffer => matrixInputBuffer.Buffer;
	private ComputeBuffer OutputBuffer => indexOutputBuffer.Buffer;

	private ComputeBufferMode MatrixBufferMode =>
		matrixBufferMode == ComputeBufferWriteMode.Immutable ? ComputeBufferMode.Immutable : ComputeBufferMode.SubUpdates;
	public int TotalCount => dimension * dimension * dimension;
	public bool UseSubUpdates => matrixBufferMode == ComputeBufferWriteMode.SubUpdates;

	[SerializeField]
	private Material mat;
	private Bounds bounds = new Bounds(Vector3.zero, Vector3.one * 10000);
	private uint[] args;
	private ComputeBuffer argsBuffer;

	private GenericNativeComputeBuffer<float3x4> matrixInputBuffer;
	private GenericNativeComputeBuffer<uint> indexOutputBuffer;
	private GenericNativeComputeBuffer<float4> colorBuffer;
	private ComputeShader appendCompute;
	private int appendComputeKernel = -1;
	private int threadCount = -1;
	private Transform camTrans;
	private TestDataGenerator dataGen;
	private DataSubset matrixBufferSubset = new DataSubset(0, 0);
	private DataSubset colorBufferSubset = new DataSubset(0, 0);
	private JobHandle currentGPUCopyJob;

	protected static int computeInputID = Shader.PropertyToID("Input");
	protected static int computeOutputID = Shader.PropertyToID("Output");
	protected static int lengthID = Shader.PropertyToID("_Length");
	protected static int cameraPosID = Shader.PropertyToID("_CameraPos");
	protected static int materialMatrixBufferID = Shader.PropertyToID("matrixBuffer");
	protected static int materialIndexBufferID = Shader.PropertyToID("indexBuffer");
	protected static int colorBufferID = Shader.PropertyToID("colorBuffer");
	protected static int maxDistanceID = Shader.PropertyToID("_MaxDistance");

	void Start()
	{
		serializedTotalCount = TotalCount;

		appendCompute = Resources.Load<ComputeShader>("InstancingAppendCompute");
		if (mat == null)
		{
			mat = new Material(InstancingShader)
			{
				//This alone doesn't actually work to set required keywords when using Graphics.DrawMeshInstancedIndirect
				enableInstancing = true
			};
		}

		appendComputeKernel = appendCompute.FindKernel("CSMain");
		appendCompute.GetKernelThreadGroupSizes(appendComputeKernel, out uint x, out _, out _);
		threadCount = (int)x;
		indexOutputBuffer = new GenericNativeComputeBuffer<uint>(new NativeArray<uint>(TotalCount, Allocator.Persistent), ComputeBufferType.Append);
		//OutputBuffer.SetCounterValue(0);

		camTrans = Camera.main.transform;
		dataGen = new TestDataGenerator(dimension);

		matrixInputBuffer = new GenericNativeComputeBuffer<float3x4>(
			new NativeArray<float3x4>(TotalCount, Allocator.Persistent), ComputeBufferType.Default, MatrixBufferMode);
		colorBuffer = new GenericNativeComputeBuffer<float4>(dataGen.colors);

		dataGen.RunMatrixJob(dimension, space, true, jobDeltaTime);
		dataGen.RunColorJob(dimension, true);
		SetMatrixBufferData();
		SetColorBufferData();

		mat.SetBuffer(materialMatrixBufferID, matrixInputBuffer.Buffer);
		mat.SetBuffer(materialIndexBufferID, indexOutputBuffer.Buffer);
		mat.SetBuffer(colorBufferID, colorBuffer.Buffer);

		InvalidateArgumentBuffer();
	}

	// This flag is used when completing the first job which copies data to the gpu.
	// It is used in LateUpdate for the first time because WaitForEndOfFrame doesn't seem to run the first frame,
	// which results in multiple different vague errors either about multiple job accessing a nativearray or that the array has been deallocated.
	private bool firstFrame = true;

	private ProfilerMarker completeDataGenerationJobMarker = new ProfilerMarker("Complete Data generation");
	void Update()
	{
		// We isolate the data generation so we clearly see other costs
		completeDataGenerationJobMarker.Begin();
		dataGen.RunMatrixJob(dimension, space, completeNow: true, jobDeltaTime);
		completeDataGenerationJobMarker.End();

		// Performance tests for pushing data to GPU
		if (TotalCount > 0 && updateAllMatrices)
			StartWriteJob();

		// Sleep 10ms for simulating workload on main thread,
		// so a job may for example run parallel to the main by completing later in the frame
		//Thread.Sleep(10);

		if (uploadGPUDataEndOfFrame && !firstFrame)
			StartCoroutine(EndOfFrameGPU_Upload());
	}

	private YieldInstruction endOfFrame = new WaitForEndOfFrame();
	private IEnumerator EndOfFrameGPU_Upload()
	{
		yield return endOfFrame;
		FinishGPU_UploadJob();
	}

	private void LateUpdate()
	{
		if (!uploadGPUDataEndOfFrame || firstFrame)
		{
			FinishGPU_UploadJob();
			firstFrame = false;
		}
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

	private void UpdateAppendCountInArgs()
	{
		//Graphics.CreateAsyncGraphicsFence
		// Copy count about 
		ComputeBuffer.CopyCount(OutputBuffer, argsBuffer, 4);
	}

	private ProfilerMarker renderMarker = new ProfilerMarker("Do Render");
	public void Render()
	{
		renderMarker.Begin();

		UpdateAppendCountInArgs();

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

	private ProfilerMarker beginGPUWriteMarker = new ProfilerMarker("BeginGPU Write");
	private Action endBufferWrite;
	private void StartWriteJob()
	{
		if (UseSubUpdates)
		{
			jobDeltaTime = Time.time - jobStartTimeSeconds;
			jobStartTimeSeconds = Time.time;
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

	private ProfilerMarker copyToGPUMarker = new ProfilerMarker("Copy data to GPU");
	private void CopyDataToGPU(ref NativeArray<float3x4> array)
	{
		copyToGPUMarker.Begin();

		switch (gpuWriteMethod)
		{
			case GPUWriteMethod.NativeArrayCopy:
				array.CopyFrom(dataGen.matrices);
				break;
			case GPUWriteMethod.ParallelForJobSet:
				currentGPUCopyJob = new ParallelCPUToGPUCopyJob<float3x4>()
				{
					src = dataGen.matrices,
					dst = array
				}.Schedule(TotalCount, Mathf.Max(1, jobInnerBatchCount));
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
				int batchCount = length / 4;
				currentGPUCopyJob = new ParallelCPUToGPUCopyJob<float3x4>()
				{
					src = dataGen.matrices,
					dst = array
				}.Schedule(length, batchCount);
				break;
			default:
				break;
		}

		copyToGPUMarker.End();
	}

	private ProfilerMarker completeCopyJobMarker = new ProfilerMarker("Completed GPU Copy Job");
	private void CompleteGPUCopyJob()
	{
		completeCopyJobMarker.Begin();
		// Finish gpu copy job
		currentGPUCopyJob.Complete();
		completeCopyJobMarker.End();
	}

	private ProfilerMarker endGPUWriteMarker = new ProfilerMarker("Finished writing GPU data");
	private void FinishGPU_UploadJob()
	{
		CompleteGPUCopyJob();

		endGPUWriteMarker.Begin();
		// Finish ComputeBuffer.BeginWrite
		endBufferWrite?.Invoke();
		endGPUWriteMarker.End();

		// ComputeBuffer.SetData
		if (!UseSubUpdates)
			SetMatrixBufferData();

		// Dispatch culling compute after we've uploaded the matrices
		DispatchCompute();
	}
	[SerializeField]
	private float maxDist = 100;
	//private GraphicsFence computeFence;
	private ProfilerMarker dispatchComputeMarker = new ProfilerMarker("Dispatch Culling Compute");
	private void DispatchCompute()
	{
		dispatchComputeMarker.Begin();
		//computeFence = Graphics.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, SynchronisationStageFlags.ComputeProcessing);

		OutputBuffer.SetCounterValue(0);
		appendCompute.SetBuffer(appendComputeKernel, computeInputID, matrixInputBuffer.Buffer); // Matrices
		appendCompute.SetBuffer(appendComputeKernel, computeOutputID, indexOutputBuffer.Buffer); // Indices
		appendCompute.SetInt(lengthID, TotalCount);
		appendCompute.SetVector(cameraPosID, camTrans.position);
		appendCompute.SetFloat(maxDistanceID, maxDist);

		dispatchX = Mathf.Min((TotalCount + threadCount - 1) / threadCount, 65535);
		appendCompute.Dispatch(appendComputeKernel, dispatchX, 1, 1);

		//Graphics.WaitOnAsyncGraphicsFence(computeFence);

		dispatchComputeMarker.End();
	}

	private ProfilerMarker pushAllMatricesMarker = new ProfilerMarker("ComputeBuffer.SetData");
	private void SetMatrixBufferData()
	{
		pushAllMatricesMarker.Begin();
		if (matrixBufferSubset.count != TotalCount - 1)
			matrixBufferSubset = new DataSubset(0, TotalCount - 1);

		if (matrixInputBuffer.SetData(matrixBufferSubset))
			mat.SetBuffer(materialMatrixBufferID, matrixInputBuffer.Buffer);

		pushAllMatricesMarker.End();
	}

	private ProfilerMarker pushAllColorsMarker = new ProfilerMarker("Update Colors and Set ComputeBuffer");
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
		matrixInputBuffer.Dispose();
		colorBuffer.Dispose();
		argsBuffer.Release();
		indexOutputBuffer.Dispose();
		dataGen.Dispose();
	}

	#region Job types

	[BurstCompile]
	public struct CPUToGPUCopyJob<T> : IJob where T : unmanaged
	{
		[ReadOnly] public NativeArray<T> src;
		[WriteOnly] public NativeArray<T> dst;

		public void Execute()
		{
			dst.CopyFrom(src);
		}
	}

	[BurstCompile]
	public struct ParallelCPUToGPUCopyJob<T> : IJobParallelFor where T : unmanaged
	{
		[ReadOnly] public NativeArray<T> src;
		[WriteOnly] public NativeArray<T> dst;

		public void Execute(int index)
		{
			dst[index] = src[index];
		}
	}

	[BurstCompile]
	public struct BatchParallelCPUToGPUCopyJob<T> : IJobParallelForBatch where T : unmanaged
	{
		[ReadOnly] public NativeArray<T> src;
		[WriteOnly] public NativeArray<T> dst;

		public void Execute(int startIndex, int count)
		{
			NativeArray<T>.Copy(src, startIndex, dst, startIndex, count);
		}
	}

	[BurstCompile]
	public struct CPUToGPUCopyForJob<T> : IJobFor where T : unmanaged
	{
		[ReadOnly] public NativeArray<T> src;
		[WriteOnly] public NativeArray<T> dst;

		public void Execute(int index)
		{
			dst[index] = src[index];
		}
	}

	#endregion
}