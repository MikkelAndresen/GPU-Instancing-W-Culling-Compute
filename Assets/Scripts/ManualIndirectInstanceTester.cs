using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
using System.Collections.Generic;
using Random = UnityEngine.Random;
using Debug = UnityEngine.Debug;
using System;
using Unity.Jobs;
using System.Collections;
using Unity.Profiling;
using System.Threading;
using Unity.Burst;

// Sort Input matrix buffer by a priority based on how often the developer thinks they will be updated.
// e.g if 10 objects will update often, put them in index 0 - 10.
// Cache all updates for any given frame in a index list, loop to find the min/max indexes to be updated,
// then create a subset to update the buffer.
// This could also we done by a job.

public class ManualIndirectInstanceTester : MonoBehaviour
{
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
	[SerializeField, SetProperty(nameof(matrixSubsetUpdateMarker))]
	private int matrixUpdateSubsetCount = 10;
	[SerializeField]
	private ComputeBufferMode matrixBufferMode = ComputeBufferMode.Dynamic;
	[SerializeField]
	private ShadowCastingMode shadowCastingMode = ShadowCastingMode.On;
	[SerializeField]
	private bool receiveShadows = true;
	[SerializeField]
	private int dispatchX;
	[SerializeField]
	private float jobDeltaTime;
	private float jobStartTimeSeconds;
	[SerializeField]
	private Shader instancingShader;
	private Shader InstancingShader => instancingShader == null ? Shader.Find("Unlit/InstancedIndirectUnlit") : instancingShader;

	private ComputeBuffer InputBuffer => matrixBuffer.Buffer;
	private ComputeBuffer outputBuffer;
	private ComputeBuffer OutputBuffer
	{
		get => outputBuffer;
		set => outputBuffer = value;
	}
	public int TotalCount => dimension * dimension * dimension;
	public bool UseComputeBufferWriteMethod => matrixBuffer != null && matrixBuffer.BufferMode == ComputeBufferMode.SubUpdates;
	private int MatrixUpdateSubsetCount
	{
		get => matrixUpdateSubsetCount;
		set
		{
			matrixUpdateSubsetCount = value;
			matrixSubsetUpdateMarker = $"Update Matrix Subset {matrixUpdateSubsetCount}";
		}
	}
	private string matrixSubsetUpdateMarker = "Update Matrix Subset";

	private Material mat;
	private Bounds bounds = new Bounds(Vector3.zero, Vector3.one * 10000);
	private uint[] args;
	private ComputeBuffer argsBuffer;

	private GenericNativeComputeBuffer<float3x4> matrixBuffer;
	private GenericNativeComputeBuffer<float4> colorBuffer;
	private ComputeShader appendCompute;
	private int appendComputeKernel = -1;
	private int threadCount = -1;
	private Transform camTrans;
	private List<float3x4> matrixBufferData = new List<float3x4>();
	private List<float4> colorBufferData = new List<float4>();
	private NativeArray<float3x4> MatrixBufferData => dataGen.matrices;
	private NativeArray<float4> ColorBufferData => dataGen.colors;
	private TestDataGenerator dataGen;

	protected static int computeInputID = Shader.PropertyToID("Input");
	protected static int computeOutputID = Shader.PropertyToID("Output");
	protected static int lengthID = Shader.PropertyToID("_Length");
	protected static int cameraPosID = Shader.PropertyToID("_CameraPos");
	protected static int materialMatrixBufferID = Shader.PropertyToID("matrixBuffer");
	protected static int materialIndexBufferID = Shader.PropertyToID("indexBuffer");
	protected static int colorBufferID = Shader.PropertyToID("colorBuffer");

	void Start()
	{
		serializedTotalCount = TotalCount;

		appendCompute = Resources.Load<ComputeShader>("InstancingAppendCompute");
		mat = new Material(InstancingShader);
		appendComputeKernel = appendCompute.FindKernel("CSMain");
		appendCompute.GetKernelThreadGroupSizes(appendComputeKernel, out uint x, out _, out _);
		threadCount = (int)x;
		camTrans = Camera.main.transform;

		dataGen = new TestDataGenerator(dimension);

		matrixBuffer = new GenericNativeComputeBuffer<float3x4>(dataGen.matrices, ComputeBufferType.Default, matrixBufferMode);
		colorBuffer = new GenericNativeComputeBuffer<float4>(dataGen.colors);

		dataGen.RunMatrixJob(dimension, space, true, jobDeltaTime);
		dataGen.RunColorJob(dimension, true);
		PushAllMatrices();
		PushAllColors();

		OutputBuffer = new ComputeBuffer(TotalCount, sizeof(uint), ComputeBufferType.Append);
		OutputBuffer.SetCounterValue(1);

		mat.SetBuffer(materialMatrixBufferID, InputBuffer);
		mat.SetBuffer(materialIndexBufferID, OutputBuffer);
		mat.SetBuffer(colorBufferID, colorBuffer.Buffer);

		InvalidateArgumentBuffer();
	}

	private JobHandle currentDataJob;
	private JobHandle currentGPUCopyJob;
	void Update()
	{
		completeDataGenerationJobMarker.Begin();
		currentDataJob = dataGen.RunMatrixJob(dimension, space, completeNow: false, jobDeltaTime);
		currentDataJob.Complete();
		completeDataGenerationJobMarker.End();

		// Performance tests for pushing data to GPU
		if (TotalCount > 0)
		{
			if (updateAllMatrices)
			{
				// Use SetData Method
				if (UseComputeBufferWriteMethod)
					StartWriteJob();
			}
		}

		Render();
		Thread.Sleep(10);
	}

	private ProfilerMarker completeDataGenerationJobMarker = new ProfilerMarker("Complete Data generation");
	private void LateUpdate()
	{
		FinishGPU_UploadJob();
	}

	private ProfilerMarker endGPUWriteMarker = new ProfilerMarker("Finished writing GPU data");
	private void FinishGPU_UploadJob()
	{
		endGPUWriteMarker.Begin();
		currentGPUCopyJob.Complete();
		endBufferWrite?.Invoke();
		endGPUWriteMarker.End();

		if (!UseComputeBufferWriteMethod)
			PushAllMatrices();
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
		//ComputeBuffer.CopyCount(OutputBuffer, argsBuffer, 4);
	}

	private ProfilerMarker renderMarker = new ProfilerMarker("Do Render");
	public void Render()
	{
		renderMarker.Begin();

		//DispatchCompute();
		//UpdateAppendCountInArgs();
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

	private GraphicsFence computeFence;
	private void DispatchCompute()
	{
		// Because of the nature of async behaviour of jobs,
		// we don't know when matrixBuffer.Buffer is setup
		if (matrixBuffer.Buffer == null)
			return;

		Profiler.BeginSample(nameof(DispatchCompute));
		//computeFence = Graphics.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, SynchronisationStageFlags.ComputeProcessing);

		OutputBuffer.SetCounterValue(0);
		appendCompute.SetBuffer(appendComputeKernel, computeInputID, matrixBuffer.Buffer);
		appendCompute.SetBuffer(appendComputeKernel, computeOutputID, OutputBuffer);
		appendCompute.SetInt(lengthID, TotalCount);
		appendCompute.SetVector(cameraPosID, camTrans.position);

		dispatchX = Mathf.Min((TotalCount + threadCount - 1) / threadCount, 65535);
		appendCompute.Dispatch(appendComputeKernel, dispatchX, 1, 1);

		//Graphics.WaitOnAsyncGraphicsFence(computeFence);

		Profiler.EndSample();
	}

	private ProfilerMarker beginGPUWriteMarker = new ProfilerMarker("BeginGPU Write");
	private Action endBufferWrite;
	private void StartWriteJob()
	{
		if (UseComputeBufferWriteMethod)
		{
			jobDeltaTime = Time.time - jobStartTimeSeconds;
			jobStartTimeSeconds = Time.time;
			beginGPUWriteMarker.Begin();
			endBufferWrite = matrixBuffer.BeginWriteMatrices<float3x4>(0, TotalCount, WriteAllMatrices);
			beginGPUWriteMarker.End();
		}
	}

	private void WriteAllMatrices(ref NativeArray<float3x4> array)
	{
		//currentGPUCopyJob = new ParallelCPUToGPUCopyJob<float3x4>()
		//{
		//	input = dataGen.matrices,
		//	output = array
		//}.Schedule(TotalCount, Mathf.Max(1, jobInnerBatchCount));

		//array.CopyFrom(dataGen.matrices);

		//currentGPUCopyJob = new CPUToGPUCopyJob<float3x4>()
		//{
		//	src = dataGen.matrices,
		//	dst = array
		//}.Schedule();

		int length = TotalCount;
		int batchCount = length / 4;
		currentGPUCopyJob = new ParallelCPUToGPUCopyJob<float3x4>()
		{
			src = dataGen.matrices,
			dst = array
		}.Schedule(length, batchCount/*jobInnerBatchCount*/);

		//currentGPUCopyJob = new CPUToGPUCopyJob<float3x4>()
		//{
		//	input = dataGen.matrices,
		//	output = array
		//}.Schedule(TotalCount, default);
	}

	private ProfilerMarker pushAllMatricesMarker = new ProfilerMarker("Update Matrices and Set ComputeBuffer");
	private void PushAllMatrices()
	{
		var subset = new DataSubset(0, TotalCount - 1);

		pushAllMatricesMarker.Begin();
		matrixBuffer.SetData(subset);
		pushAllMatricesMarker.End();
	}

	private ProfilerMarker pushAllColorsMarker = new ProfilerMarker("Update Colors and Set ComputeBuffer");
	private void PushAllColors()
	{
		var subset = new DataSubset(0, TotalCount - 1);

		pushAllColorsMarker.Begin();
		colorBuffer.SetData(subset);
		pushAllColorsMarker.Begin();
	}

	private void OnDestroy()
	{
		matrixBuffer.Dispose();
		colorBuffer.Dispose();
		argsBuffer.Release();
		OutputBuffer.Release();
		dataGen.Dispose();
	}

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
		[ReadOnly]public NativeArray<T> src;
		[WriteOnly]public NativeArray<T> dst;

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
}