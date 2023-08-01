using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Jobs;
using Plane = MathUtil.Plane;

// TODO Generate bounds from sizes and matrices or transforms
// TODO Support static use of matrices, do we want to upload matrices using culling every frame despite them being static 
// or do we change the shader to use indexed matrices so that we only upload the indices instead?

public class FrustrumFilterTransformJobSystem : MonoBehaviour
{
	[SerializeField] private int transformGatherBatchSize = 10;
	[SerializeField] private int indexingBatchCount = 10;
	[SerializeField] private int writeBatchCount = 10;
	[SerializeField] private float3 defaultBoundSize = 1;

	// [SerializeField, Tooltip("When this is true, we only upload the matrices from the transforms at startup")]
	// private bool staticTransforms;

	private TransformAccessArray transformArray;
	private NativeArray<float3x4> matrices;
	private ComputeBuffer matrixBuffer;
	private NativeArray<Plane> frustum;
	private NativeArray<float3> boundSizes;
	private UnityEngine.Plane[] unityFrustum;
	private Camera mainCam;
	private JobHandle handle;
	private NativeCollectionExtensions.CopyHandler<float3x4, InFrustum> copyHandler;
	
	private void Start()
	{
		mainCam = Camera.main;
		transformArray = new TransformAccessArray(GetComponentsInChildren<Transform>());
		matrices = new NativeArray<float3x4>(transformArray.length, Allocator.Persistent);
		matrixBuffer = new ComputeBuffer(matrices.Length, UnsafeUtility.SizeOf(typeof(float3x4)), ComputeBufferType.Default,
			ComputeBufferMode.SubUpdates);
		boundSizes = new NativeArray<float3>(matrices.Length, Allocator.Persistent);

		GenerateDefaultBoundSizes();

		copyHandler = new NativeCollectionExtensions.CopyHandler<float3x4, InFrustum>(default, default, indexingBatchCount, writeBatchCount, default);
	}

	private void OnDestroy()
	{
		transformArray.Dispose();
		matrices.Dispose();
		matrixBuffer.Dispose();
		frustum.Dispose();
		boundSizes.Dispose();
		copyHandler.Dispose();
	}

	private void Update()
	{
		// if(staticTransforms)
		// 	return;
		
		InvalidateFrustumPlaneCache();

		GatherMatricesTransformJob matrixGather = new GatherMatricesTransformJob(matrices);
		InFrustum frustumChecker = new InFrustum()
		{
			frustum = frustum,
			positions = matrices.Slice().SliceConvert<float3>(),
			boundSizes = boundSizes
		};

		copyHandler.validator = frustumChecker;
		copyHandler.indexingBatchCount = indexingBatchCount;
		copyHandler.writeBatchCount = writeBatchCount;
		copyHandler.src = matrices;
		copyHandler.dst = matrixBuffer.BeginWrite<float3x4>(0, matrices.Length);
		
		// Possible job dependency conflict with dst because it is being written to by first job then second job gets created? 
		handle = matrixGather.ScheduleReadOnlyByRef(transformArray, transformGatherBatchSize);
		handle = copyHandler.IfCopyToParallel(handle);
		// handle = matrices.IfCopyToParallel(dst, out counter, indexingBatchCount, writeBatchCount, handle, indices, counts, frustumChecker);
	}
	
	private void LateUpdate()
	{
		handle.Complete();
		matrixBuffer.EndWrite<float3x4>(copyHandler.CountCopied);
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

	public struct InFrustum : IValidator<float3x4>
	{
		[ReadOnly] public NativeArray<Plane> frustum;

		[ReadOnly, NativeDisableParallelForRestriction]
		public NativeSlice<float3> boundSizes;

		[ReadOnly] public NativeSlice<float3> positions;
		// TODO Make a diff check to see if the matrix is different and perhaps return false if not?

		// Generate bound based on position and current size array, then compare it against frustum
		public bool Validate(int index, float3x4 element) => new MathUtil.AABB(positions[index], boundSizes[index]).IsBoundsInFrustum(frustum);
	}
}