using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Rendering;
using System;
using Unity.Jobs;
using Unity.Profiling;
using CommonJobs;
using Plane = MathUtil.Plane;

// Sort Input matrix buffer by a priority based on how often the developer thinks they will be updated.
// e.g if 10 objects will update often, put them in index 0 - 10.
// Cache all updates for any given frame in a index list, loop to find the min/max indexes to be updated,
// then create a subset to update the buffer.
// This could also we done by a job.

public class DrawInstancedJobsCulled : MonoBehaviour
{
	[SerializeField]
	private Mesh mesh;
	[SerializeField]
	private int dimension = 10;
	[SerializeField]
	private int serializedTotalCount;
	[SerializeField]
	private int serializedCurrentCount;
	//[SerializeField] // This was used to check that the counter values make sense for a very odd bug related to the burst compiler
	//private int[] pastCounterValues = new int[60];
	//private int renderFrameCounter;
	[SerializeField]
	private float space = 1.5f;

	[SerializeField]
	private ShadowCastingMode shadowCastingMode = ShadowCastingMode.On;
	[SerializeField]
	private bool receiveShadows = true;

	[SerializeField]
	private Shader instancingShader;
	private Shader InstancingShader => instancingShader == null ? Shader.Find("Unlit/InstancedIndirectUnlit") : instancingShader;

	public int TotalCount => dimension * dimension * dimension;
	/// <summary>
	/// This represents how many elements are being rendered, it cannot be public because it is used by jobs
	/// </summary>
	public int CurrentRenderCount { get; private set; }

	private Bounds bounds = new Bounds(Vector3.zero, Vector3.one * 10000);
	private Material mat;

	private GenericNativeComputeBuffer<float3x4> matrixInputBuffer;
	private NativeList<int> matrixIndices;
	private ComputeBuffer indexBuffer;
	private GenericNativeComputeBuffer<float4> colorBuffer;
	private float3 boundSize = new float3(1, 1, 1);

	private TestDataGenerator dataGen;
	private DataSubset matrixBufferSubset = new DataSubset(0, 0);
	private DataSubset indexBufferSubset = new DataSubset(0, 0);
	private DataSubset colorBufferSubset = new DataSubset(0, 0);
	private readonly UnityEngine.Plane[] frustumPlanes = new UnityEngine.Plane[6];
	private NativeArray<MathUtil.Plane> nativeFrustumPlanes;
	[SerializeField]
	private MathUtil.Plane[] serializedFrustum = new Plane[6];
	private JobHandle cullJobHandle;
	private Camera mainCam;

	protected static int materialMatrixBufferID = Shader.PropertyToID("matrixBuffer");
	protected static int materialIndexBufferID = Shader.PropertyToID("indexBuffer");
	protected static int colorBufferID = Shader.PropertyToID("colorBuffer");

	void Start()
	{
		serializedTotalCount = TotalCount;
		mainCam = Camera.main;

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

		dataGen = new TestDataGenerator(dimension, transform);
		nativeFrustumPlanes = new NativeArray<Plane>(6, Allocator.Persistent);
		matrixIndices = new NativeList<int>(TotalCount, Allocator.Persistent);
		indexBuffer = new ComputeBuffer(TotalCount, sizeof(int));
		colorBuffer = new GenericNativeComputeBuffer<float4>(dataGen.colors);
		boundSize = mesh.bounds.size;

		InvalidateMatrixBuffer();

		// Set all inital data
		dataGen.RunMatrixJob(dimension, space, true, Time.deltaTime);
		dataGen.RunColorJob(dimension, true);
		SetMatrixBufferData();
		SetColorBufferData();

		mat.SetBuffer(materialIndexBufferID, indexBuffer);
		mat.SetBuffer(colorBufferID, colorBuffer.Buffer);

		RenderPipelineManager.beginFrameRendering += RenderPipelineManager_beginFrameRendering;
		RenderPipelineManager.beginCameraRendering += RenderPipelineManager_beginCameraRendering; ;
	}

	private void InvalidateMatrixBuffer()
	{
		if (matrixInputBuffer != null)
			matrixInputBuffer.Dispose();

		matrixInputBuffer = new GenericNativeComputeBuffer<float3x4>(
			new NativeArray<float3x4>(TotalCount, Allocator.Persistent), ComputeBufferType.Default, ComputeBufferMode.SubUpdates); // Should be subupdates
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

		// We isolate the data generation so we clearly see other costs
		completeDataGenerationJobMarker.Begin();
		dataGen.RunMatrixJob(dimension, space, completeNow: true, Time.deltaTime);
		completeDataGenerationJobMarker.End();

		// Performance tests for pushing data to GPU
		if (TotalCount > 0)
		{
			InvalidateFrustumPlaneCache();
			StartWriteJob();
		}
	}

	private void ResetIndexBufferCounter() => indexBuffer.SetCounterValue(0);

	private readonly ProfilerMarker renderMarker = new ProfilerMarker("Render");

	public void Render(Camera cam)
	{
		//Debug.Log("Do Render " + Time.frameCount);

		renderMarker.Begin();
		//renderFrameCounter++;
		//if (renderFrameCounter >= pastCounterValues.Length)
		//	renderFrameCounter = 0;

		if (CurrentRenderCount == 0)
		{
			renderMarker.End();
			serializedCurrentCount = 0;
			return;
		}

		serializedCurrentCount = CurrentRenderCount;
		//pastCounterValues[renderFrameCounter] = serializedCurrentCount;

		Graphics.DrawMeshInstancedProcedural(
			mesh,
			submeshIndex: 0,
			mat,
			bounds,
			count: CurrentRenderCount,
			properties: null,
			shadowCastingMode,
			receiveShadows,
			gameObject.layer,
			camera: cam,
			LightProbeUsage.BlendProbes,
			lightProbeProxyVolume: null
		);

		renderMarker.End();
	}

	private void RenderPipelineManager_beginFrameRendering(ScriptableRenderContext arg1, Camera[] arg2) => FinishGPU_UploadJob();

	private void RenderPipelineManager_beginCameraRendering(ScriptableRenderContext arg1, Camera cam) => Render(cam);

	private readonly ProfilerMarker beginGPUWriteMarker = new ProfilerMarker("BeginGPU Write");
	private Action endMatrixBufferWrite, endIndexBufferWrite;
	private void StartWriteJob()
	{
		CurrentRenderCount = 0;

		beginGPUWriteMarker.Begin();
		if (matrixInputBuffer.BufferBeingWritten)
		{
			beginGPUWriteMarker.End();
			return;
		}

		//Debug.Log("Start GPU Write Job" + Time.frameCount);

		var (endMatrixWrite, matrices) = matrixInputBuffer.BeginWriteMatrices(0, TotalCount);
		endMatrixBufferWrite = endMatrixWrite;

		CullAndCopyDataToGPU(matrices);

		beginGPUWriteMarker.End();
	}

	[SerializeField]
	private int innerBatchCount = 10;
	private CPUToGPUCopyAndCullFilterJob cullJob;
	private readonly ProfilerMarker copyToGPUMarker = new ProfilerMarker("Copy data to GPU");
	private void CullAndCopyDataToGPU(NativeArray<float3x4> matrices)
	{
		copyToGPUMarker.Begin();

		// Reset counter
		matrixIndices.Clear();
		cullJob = new CPUToGPUCopyAndCullFilterJob()
		{
			srcMatrices = dataGen.matrices,
			dstMatrices = matrices,
			frustum = nativeFrustumPlanes,
			boundSize = boundSize
			
		};
		cullJobHandle = cullJob.ScheduleAppend(matrixIndices, TotalCount, innerBatchCount);

		copyToGPUMarker.End();
	}

	private readonly ProfilerMarker completeCopyJobMarker = new ProfilerMarker("Completed GPU Copy Job");
	private void CompleteGPUCopyJob()
	{
		completeCopyJobMarker.Begin();

		// Finish gpu copy job
		cullJobHandle.Complete();
		if (matrixIndices.IsCreated)
			CurrentRenderCount = matrixIndices.Length;

		completeCopyJobMarker.End();
	}

	private readonly ProfilerMarker endGPUWriteMarker = new ProfilerMarker("Finished writing GPU data");
	private void FinishGPU_UploadJob()
	{
		CompleteGPUCopyJob();

		endGPUWriteMarker.Begin();
		// Finish matrix buffer BeginWrite
		endMatrixBufferWrite?.Invoke();
		// Finish index buffer BeginWrite
		endIndexBufferWrite?.Invoke();

		// Set to null so there is no chance of reuse, otherwise we get an exception about the array being deallocated
		endMatrixBufferWrite = null;
		endIndexBufferWrite = null;

		SetIndexBufferData();

		endGPUWriteMarker.End();
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

	private readonly ProfilerMarker pushAllIndicesMarker = new ProfilerMarker("ComputeBuffer.SetData");
	private void SetIndexBufferData()
	{
		pushAllIndicesMarker.Begin();
		if (indexBufferSubset.count != CurrentRenderCount)
			indexBufferSubset = new DataSubset(0, CurrentRenderCount);

		indexBuffer.SetData(matrixIndices.AsArray(), 0, 0, matrixIndices.Length);

		pushAllIndicesMarker.End();
	}

	private readonly ProfilerMarker pushAllColorsMarker = new ProfilerMarker("Update Colors and Set ComputeBuffer");
	private void SetColorBufferData()
	{
		pushAllColorsMarker.Begin();
		if (colorBufferSubset.count != TotalCount - 1)
			colorBufferSubset = new DataSubset(0, TotalCount - 1);
		colorBuffer.SetData(colorBufferSubset);
		pushAllColorsMarker.End();
	}

	private readonly ProfilerMarker invalidateFrustumPlanesMarker = new ProfilerMarker("Invalidate Frustum planes cache");

	private void InvalidateFrustumPlaneCache()
	{
		invalidateFrustumPlanesMarker.Begin();

		GeometryUtility.CalculateFrustumPlanes(mainCam, frustumPlanes);
		for (int i = 0; i < frustumPlanes.Length; i++)
		{
			nativeFrustumPlanes[i] = new Plane(frustumPlanes[i]);
			serializedFrustum[i] = nativeFrustumPlanes[i];
		}

		invalidateFrustumPlanesMarker.End();
	}

	private void OnDestroy()
	{
		// We wanna finish the jobs here in case it's still running
		// because they depend on stuff we dispose below
		FinishGPU_UploadJob();

		matrixInputBuffer.Dispose();
		colorBuffer.Dispose();
		indexBuffer.Dispose();
		dataGen.Dispose();
		matrixIndices.Dispose();
		nativeFrustumPlanes.Dispose();
		RenderPipelineManager.beginFrameRendering -= RenderPipelineManager_beginFrameRendering;
		RenderPipelineManager.beginCameraRendering -= RenderPipelineManager_beginCameraRendering;
	}
}