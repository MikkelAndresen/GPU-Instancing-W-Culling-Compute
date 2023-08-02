using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using static MathUtil;
using Plane = MathUtil.Plane;

namespace CommonJobs
{
	// This test class shows how we can use an interface to implement a schedule method and create anonomoys job types,
	// which could for example be stored in an array alongside other job types
	//public class Test
	//{
	//	private interface IJobScheduler
	//	{
	//		JobHandle Schedule(int arrayLength, int innerloopBatchCount, JobHandle dependsOn = default);
	//	}

	//	private struct TestJob : IJobParallelFor, IJobScheduler
	//	{
	//		public void Execute(int index)
	//		{
	//			throw new System.NotImplementedException();
	//		}

	//		public JobHandle Schedule(int arrayLength, int innerloopBatchCount, JobHandle dependsOn = default) =>
	//			Schedule(arrayLength, innerloopBatchCount, dependsOn);
	//	}
	//}

	[BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
	public struct CPUToGPUCopyAndCullJob : IJobFor
	{
		[ReadOnly]
		public NativeArray<Plane> frustum;
		[ReadOnly]
		public NativeSlice<AABB> bounds;
		[ReadOnly]
		public NativeSlice<float3x4> srcMatrices;
		[WriteOnly]
		public NativeSlice<float3x4> dstMatrices;
		public NativeList<int>.ParallelWriter indices; // Need to be initialized to same capacity as input matrix array

		public CPUToGPUCopyAndCullJob(
			NativeList<int>.ParallelWriter indices,
			NativeSlice<AABB> bounds,
			NativeSlice<float3x4> srcMatrices,
			NativeSlice<float3x4> dstMatrices,
			NativeArray<Plane> frustum)
		{
			this.frustum = frustum;
			this.bounds = bounds;
			this.srcMatrices = srcMatrices;
			this.dstMatrices = dstMatrices;
			this.indices = indices;
		}

		public void Execute(int indexOfMatrix)
		{
			// This seemingly has issues with burst
			if (!MathUtil.IsPointInFrustum(frustum, srcMatrices[indexOfMatrix].c3))
				return;

			// Assign the matrix value to the GPU data
			dstMatrices[indexOfMatrix] = srcMatrices[indexOfMatrix];

			// Assign the matrix index to the index array for the shader to read
			indices.AddNoResize(indexOfMatrix);
		}
	}

	[BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
	public struct CPUToGPUCopyAndCullFilterJob : IJobFilter
	{
		[ReadOnly] public NativeArray<float3x4> srcMatrices;
		[ReadOnly] public NativeArray<AABB> bounds;
		[WriteOnly] public NativeArray<float3x4> dstMatrices;
		[ReadOnly]
		public NativeArray<Plane> frustum;

		public bool Execute(int index)
		{
			bool valid = bounds[index].IsBoundsInFrustum(frustum);

			// Assign the matrix value to the GPU data
			if (valid)
				dstMatrices[index] = srcMatrices[index];
			return valid;
		}
	}

	[BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
	public struct CPUToGPUCopyJob<T> : IJob where T : unmanaged
	{
		[ReadOnly] public NativeArray<T> src;
		[WriteOnly] public NativeArray<T> dst;

		public void Execute()
		{
			dst.CopyFrom(src);
		}
	}

	[BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
	public struct ParallelCPUToGPUCopyJob<T> : IJobParallelFor where T : unmanaged
	{
		[ReadOnly] public NativeArray<T> src;
		[WriteOnly] public NativeArray<T> dst;

		public void Execute(int index)
		{
			dst[index] = src[index];
		}
	}

	[BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
	public struct BatchParallelCPUToGPUCopyJob<T> : IJobParallelForBatch where T : unmanaged
	{
		[ReadOnly] public NativeArray<T> src;
		[WriteOnly] public NativeArray<T> dst;

		public void Execute(int startIndex, int count)
		{
			NativeArray<T>.Copy(src, startIndex, dst, startIndex, count);
		}
	}

	[BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
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