using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Jobs;
using Plane = MathUtil.Plane;
using static NativeCollectionExtensions;

// TODO Generate bounds from sizes and matrices or transforms
// TODO Support static use of matrices, do we want to upload matrices using culling every frame despite them being static 
// or do we change the shader to use indexed matrices so that we only upload the indices instead?

/// <summary>
/// Allocated a transform array and a matrix array which the transforms are copied into each frame.
/// Then filteres the matrix array using the <see cref="InFrustum"/> validation struct.
/// Lastly copies filtered data into a compute buffer using <see cref="copyHandler"/>.
/// </summary>
[DefaultExecutionOrder(-1)]
public class FrustrumFilterTransformJobSystem : MonoBehaviour
{
	[SerializeField] private int transformGatherBatchSize = 10;
	[SerializeField] private int indexingBatchCount = 10;
	[SerializeField] private int writeBatchCount = 10;
	[SerializeField] private float3 defaultBoundSize = 1;

	[SerializeField, Tooltip("When true, will always complete the filter job in late update, otherwise will need to have it's complete function called manually")]
	private bool autoCompleteInLateUpdate = true;

	public bool AutoCompleteInLateUpdate
	{
		get => autoCompleteInLateUpdate;
		set => autoCompleteInLateUpdate = value;
	}

	public ComputeBuffer MatrixBuffer => matrixBuffer ??= new ComputeBuffer(TransformCount, sizeof(float) * 3 * 4)
	{
		name = "Matrix Buffer"
	};

	public int FilteredCount => CopyHandler.CountCopied;
	public NativeArray<float3x4> Matrices => CopyHandler.src;
	public bool FilterJobRunning { get; private set; }

	private NativeCollectionExtensions.CopyHandler<float3x4, InFrustum> copyHandler;
	private NativeCollectionExtensions.CopyHandler<float3x4, InFrustum> CopyHandler => copyHandler ??= new CopyHandler<float3x4, InFrustum>(TransformCount, default, indexingBatchCount, writeBatchCount, default);

	private TransformAccessArray transformArray;
	private ComputeBuffer matrixBuffer;
	private NativeArray<Plane> frustum;
	private NativeArray<float3> boundSizes;
	private readonly UnityEngine.Plane[] unityFrustum = new UnityEngine.Plane[6];
	private Camera mainCam;
	private JobHandle handle;

	// TODO Make dynamic so we can add/remove externally
	private int TransformCount => transform.childCount;

	private void Awake()
	{
		mainCam = Camera.main;
		frustum = new NativeArray<Plane>(6, Allocator.Persistent);
		GatherChildTransforms();

		if (transformArray.length == 0)
		{
			enabled = false;
			return;
		}

		matrixBuffer = new ComputeBuffer(Matrices.Length, UnsafeUtility.SizeOf(typeof(float3x4)), ComputeBufferType.Default,
			ComputeBufferMode.SubUpdates);
		matrixBuffer.name = "Instance Matrix buffer";

		boundSizes = new NativeArray<float3>(Matrices.Length, Allocator.Persistent);

		GenerateDefaultBoundSizes();
	}

	private void GatherChildTransforms()
	{
		var allTransforms = GetComponentsInChildren<Transform>();
		transformArray = new TransformAccessArray(allTransforms.Length);
		foreach (var t in allTransforms)
			if (t != transform && t.gameObject.activeInHierarchy)
				transformArray.Add(t);
	}

	private unsafe void Update()
	{
		if (FilterJobRunning)
			CompleteFilterJob();

		InvalidateFrustumPlaneCache();

		GatherMatricesTransformJob matrixGather = new GatherMatricesTransformJob(Matrices);
		CopyHandler.validator = new InFrustum()
		{
			frustum = frustum,
			boundSizes = boundSizes
		};
		CopyHandler.indexingBatchCount = indexingBatchCount;
		CopyHandler.writeBatchCount = writeBatchCount;
		CopyHandler.dst = matrixBuffer.BeginWrite<float3x4>(0, Matrices.Length);
		float3x4* dstPtr = (float3x4*)CopyHandler.dst.GetUnsafeReadOnlyPtr();

		// Possible job dependency conflict with dst because it is being written to by first job then second job gets created? 
		FilterJobRunning = true;
		handle = matrixGather.ScheduleReadOnlyByRef(transformArray, transformGatherBatchSize);
		handle = CopyHandler.IfCopyToParallelUnsafe(dstPtr, handle);
	}

	private void LateUpdate()
	{
		if (autoCompleteInLateUpdate)
			CompleteFilterJob();
	}

	private static readonly ProfilerMarker completeFrustumFilterJobMarker = new ProfilerMarker("CompleteFrustumFilterJob");

	public void CompleteFilterJob()
	{
		if (!FilterJobRunning)
			return;

		completeFrustumFilterJobMarker.Begin();
		handle.Complete();
		matrixBuffer.EndWrite<float3x4>(CopyHandler.CountCopied);
		FilterJobRunning = false;
		completeFrustumFilterJobMarker.End();
	}

	private readonly ProfilerMarker invalidateFrustumPlanesMarker = new ProfilerMarker("Invalidate Frustum planes cache");

	private void InvalidateFrustumPlaneCache()
	{
		invalidateFrustumPlanesMarker.Begin();

		GeometryUtility.CalculateFrustumPlanes(mainCam, unityFrustum);
		for (int i = 0; i < unityFrustum.Length; i++)
			frustum[i] = new Plane(unityFrustum[i]);

		invalidateFrustumPlanesMarker.End();
	}

	// TODO Make this use mesh bounds instead and at some point make it settable using double buffering to avoid job conflicts
	private void GenerateDefaultBoundSizes()
	{
		for (int i = 0; i < boundSizes.Length; i++)
			boundSizes[i] = defaultBoundSize;
	}

	private void OnDestroy()
	{
		transformArray.Dispose();
		matrixBuffer.Dispose();
		frustum.Dispose();
		boundSizes.Dispose();
		CopyHandler.Dispose();
	}

	public struct InFrustum : IValidator<float3x4>
	{
		[ReadOnly] public NativeArray<Plane> frustum;

		[ReadOnly, NativeDisableParallelForRestriction]
		public NativeSlice<float3> boundSizes;

		// TODO Make a diff check to see if the matrix is different and perhaps return false if not?

		// Generate bound based on position and current size array, then compare it against frustum
		public bool Validate(int index, float3x4 element) => new MathUtil.AABB(element.c3, boundSizes[index]).IsBoundsInFrustum(frustum);
		// public bool Validate(int index, float3x4 element) => MathUtil.IsPointInFrustum(frustum, element.c3);
	}
}