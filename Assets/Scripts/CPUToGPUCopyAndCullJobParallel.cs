using CommonJobs;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using static MathUtil;

[BurstCompile]
public struct CPUToGPUCopyAndCullJobParallel : IJobParallelFor
{
	public NativeList<int>.ParallelWriter indices; // Need to be initialized to same capacity as input matrix array
	[ReadOnly]
	public NativeArray<Plane> frustum;
	[ReadOnly, NativeDisableParallelForRestriction]
	private NativeSlice<AABB4> bounds;
	[ReadOnly, NativeDisableParallelForRestriction]
	public NativeArray<float3x4> srcMatrices;
	[WriteOnly, NativeDisableParallelForRestriction]
	public NativeArray<float3x4> dstMatrices;

	public CPUToGPUCopyAndCullJobParallel(
		NativeList<int> indices,
		NativeArray<AABB> bounds,
		NativeArray<float3x4> srcMatrices,
		NativeArray<float3x4> dstMatrices,
		NativeArray<Plane> frustum)
	{
		this.indices = indices.AsParallelWriter();

		this.bounds = bounds.Slice().SliceConvert<AABB4>();
		this.srcMatrices = srcMatrices;
		this.dstMatrices = dstMatrices;

		this.frustum = frustum;
	}

	// TODO Can we use math.compress to fill data into the dstMatrix?
	public void Execute(int index) => VectorizedWriter(index);

	// This takes about 4.8ms on a 5950x
	private unsafe void VectorizedWriter(int index)
	{
		int k = index * 4;
		bool4 result = bounds[index].IsBoundsInFrustum(frustum);
		//bool4 result = default;
		//for (int i = 0; i < 4; i++)
		//	result[i] = boundsSlice[index][i].IsBoundsInFrustum(frustum);

		int4 tmp = default;
		int count = 0;
		for (int i = 0; i < 4; i++)
		{
			if (result[i])
			{
				dstMatrices[k + i] = srcMatrices[k + i];
				tmp[count] = k + i;
				count++;
			}
		}
		indices.AddRangeNoResize(&tmp, count);
	}

	// This takes about 14.5ms on a 5950x
	private void SimpleWriter()
	{
		//if (bounds[index].IsBoundsInFrustum(frustum))
		//{
		//	dstMatrices[index] = srcMatrices[index];
		//	indices.AddNoResize(index);
		//}
	}

	public static JobHandle Schedule(
		NativeList<int> indices,
		NativeArray<AABB> bounds,
		NativeArray<float3x4> srcMatrices,
		NativeArray<float3x4> dstMatrices,
		NativeArray<Plane> frustum,
		out CPUToGPUCopyAndCullJobParallel mainJob,
		out CPUToGPUCopyAndCullJob remainderJob,
		int innerBatchLoopCount = 64, JobHandle dependsOn = default)
	{
		mainJob = new CPUToGPUCopyAndCullJobParallel(
			indices: indices,
			bounds: bounds,
			srcMatrices: srcMatrices,
			dstMatrices: dstMatrices,
			frustum: frustum);

		var handle = mainJob.Schedule(mainJob.bounds.Length, innerBatchLoopCount, dependsOn);
		//var handle = mainJob.Schedule();

		remainderJob = default;
		int totalLength = srcMatrices.Length;
		int remainder = totalLength % 4;
		int remainderStartIndex = totalLength - remainder;
		if (remainder != 0)
		{
			var slicedRemainderBounds = bounds.Slice(remainderStartIndex, remainder);
			var srcMatricesRemainderSlice = srcMatrices.Slice(remainderStartIndex, remainder);
			var dstMatricesRemainderSlice = dstMatrices.Slice(remainderStartIndex, remainder);

			remainderJob = new CPUToGPUCopyAndCullJob(
				indices.AsParallelWriter(),
				slicedRemainderBounds,
				srcMatricesRemainderSlice,
				dstMatricesRemainderSlice,
				frustum);

			handle = remainderJob.Schedule(remainder, handle);
		}

		return handle;
	}
}