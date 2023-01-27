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
		[ReadOnly] public NativeArray<float3x4> srcMatrices;
		[WriteOnly] public NativeArray<float3x4> dstMatrices;
		[ReadOnly]
		public NativeArray<Plane> frustum;
		[WriteOnly] public NativeArray<uint> indices;
		public NativeReference<int> counter;

		public void Execute(int indexOfMatrix)
		{
			// This seemingly has issues with burst
			if (!MathUtil.IsPointInFrustum(frustum, srcMatrices[indexOfMatrix].c3))
				return;

			// Assign the matrix value to the GPU data
			dstMatrices[indexOfMatrix] = srcMatrices[indexOfMatrix];

			// Assign the matrix index to the index array for the shader to read
			indices[counter.Value] = (uint)indexOfMatrix;

			// Increment index counter, we use this because we need to keep track of how many matrices we wanna render
			counter.Value++;
		}
	}

	[BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
	public struct CPUToGPUCopyAndCullFilterJob : IJobParallelForFilter
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