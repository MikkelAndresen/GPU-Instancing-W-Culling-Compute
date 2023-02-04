using CommonJobs;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Profiling;
using UnityEngine;
using static MathUtil;

public interface IValidator<T> where T : unmanaged
{
	bool Validate(T element);
}

#region Parallel Indexing, single copy

/// <summary>
/// This job is meant to pack booleans into <see cref="indices"/>.
/// Then you can use <see cref="ConditionalCopyMergeJob{T}"/> to write to a destination array based on the <see cref="indices"/> array.
/// </summary>
/// <typeparam name="T"></typeparam>
[BurstCompile, BurstCompatible]
public struct ConditionIndexingJob<T, M> : IJobParallelFor where T : unmanaged where M : IValidator<T>
{
	[ReadOnly]
	public M del;
	[ReadOnly]
	public NativeArray<T> src;
	[WriteOnly]
	public NativeArray<BitField64> indices;
	private static readonly ProfilerMarker conditionIndexingJobMarker = new ProfilerMarker(nameof(ConditionIndexingJob<T, M>));

	public ConditionIndexingJob(NativeArray<T> src, NativeArray<BitField64> indices, M del = default)
	{
		this.src = src;
		this.indices = indices;
		this.del = del;
	}

	public void Execute(int index)
	{
		//conditionIndexingJobMarker.Begin();

		BitField64 bits = new BitField64(0);
		//byte* bitsPtr = (byte*)&bits.Value;
		//const int iterations = 64 / 4;
		int dataIndex = index * 64;

		for (int i = 0; i < 64; i++)
			bits.SetBits(i, del.Validate(src[dataIndex + i]));
		indices[index] = bits;

		//conditionIndexingJobMarker.End();
	}

	private static readonly ProfilerMarker remainderJobMarker = new ProfilerMarker(nameof(RemainderJob));
	[BurstCompile, BurstCompatible]
	private struct RemainderJob : IJob
	{
		[ReadOnly]
		public M del;
		[ReadOnly]
		public NativeArray<T> src;
		[WriteOnly]
		public NativeArray<BitField64> indices;
		private BitField64 bits;
		public void Execute()
		{
			//remainderJobMarker.Begin();

			int remainderCount = src.Length % 64;
			int dataStartIndex = src.Length - remainderCount;
			bits.Clear();

			for (int i = 0; i < remainderCount; i++)
				bits.SetBits(i, del.Validate(src[dataStartIndex + i]));

			indices[indices.Length - 1] = bits;

			//remainderJobMarker.End();
		}
	}

	public static JobHandle Schedule(
		NativeArray<T> src,
		NativeArray<BitField64> indices,
		out ConditionIndexingJob<T, M> job,
		int innerBatchCount = 10,
		M del = default)
	{
		int remainder = src.Length % 64;
		// The job only supports writing whole 64 bit batches, so we floor here and then run the remainder elsewhere
		int length = (int)math.floor(src.Length / 64f);
		job = new ConditionIndexingJob<T, M>(src, indices);

		var handle = job.Schedule(length, innerBatchCount);
		if (remainder > 0)
			handle = new RemainderJob()
			{
				src = src,
				indices = indices,
				del = del
			}.Schedule(handle);

		return handle;
	}
}

[BurstCompile, BurstCompatible]
public unsafe struct ConditionalCopyMergeJob<T> : IJobFor where T : unmanaged
{
	[ReadOnly, NativeDisableParallelForRestriction]
	public NativeArray<T> src;
	[WriteOnly, NativeDisableParallelForRestriction]
	public NativeArray<T> dst;

	//[NativeDisableUnsafePtrRestriction, ReadOnly]
	//private readonly BitField64* bitsPtr;
	private NativeArray<BitField64> indices;
	private readonly int indicesLength;
	[NativeDisableUnsafePtrRestriction, ReadOnly]
	private readonly T* srcPtr;
	[NativeDisableUnsafePtrRestriction, ReadOnly]
	private readonly T* dstPtr;
	private long sizeOfT;
	private static readonly ProfilerMarker conditionalCopyMergeJobMarker = new ProfilerMarker(nameof(ConditionalCopyMergeJob<T>));

	private NativeReference<int> counter;

	public unsafe ConditionalCopyMergeJob(
		NativeArray<T> src,
		NativeArray<T> dst,
		NativeArray<BitField64> indices,
		NativeReference<int> count)
	{
		this.src = src;
		this.dst = dst;
		indicesLength = indices.Length;
		this.counter = count;
		//bitsPtr = (BitField64*)indices.GetUnsafeReadOnlyPtr();
		this.indices = indices;
		srcPtr = (T*)src.GetUnsafeReadOnlyPtr();
		dstPtr = (T*)dst.GetUnsafePtr();
		sizeOfT = UnsafeUtility.SizeOf(typeof(T));
		sequentialIndicesStartIndex = default;
	}

	public void Execute(int index)
	{
		ExecuteNoBatches(index);
		//ExecuteBatched(index);
	}

	private void ExecuteNoBatches(int index)
	{
		//conditionalCopyMergeJobMarker.Begin();

		int countSoFar = counter.Value;
		int srcStartIndex = index * 64;

		ulong n = indices[index].Value;
		int t = 0;
		//int popCount = math.countbits(n);
		//for (int i = 0; i < popCount; i++)
		//{
		//	int tzcnt = math.tzcnt(n);
		//	t += tzcnt;
		//	//Debug.Log($"Writing from index {srcStartIndex + t + i}");
		//	// MemCpy is slower for some reason
		//	//UnsafeUtility.MemCpy(dstPtr + countSoFar, srcPtr + srcStartIndex + t + i, sizeOfT);
		//	dst[countSoFar] = src[srcStartIndex + t + i];
		//	i++;
		//	countSoFar++;
		//	n >>= tzcnt + 1;
		//}
		//if (math.countbits(n) == 64)
		//{
		//	// This is more expensive for some reason than the while loop
		//	UnsafeUtility.MemCpy(dstPtr + countSoFar, srcPtr + srcStartIndex + t + 64, sizeOfT);
		//	counter.Value += 64;
		//}
		//else
		//{
		int i = 0;
		while (n != 0)
		{
			int tzcnt = math.tzcnt(n);
			t += tzcnt;
			//Debug.Log($"Writing from index {srcStartIndex + t + i}");

			// MemCpy is slower for some reason
			//UnsafeUtility.MemCpy(dstPtr + countSoFar, srcPtr + srcStartIndex + t + i, sizeOfT);

			dst[countSoFar] = src[srcStartIndex + t + i];
			countSoFar++;
			i++;

			n >>= tzcnt + 1;
		}

		counter.Value = countSoFar;
		//}

		//conditionalCopyMergeJobMarker.End();
	}

	int sequentialIndicesStartIndex;
	public void ExecuteBatched(int index)
	{
		//conditionalCopyMergeJobMarker.Begin();

		int countSoFar = counter.Value;
		int srcStartIndex = index * 64;

		ulong n = indices[index].Value;
		int cumulativeTzcnt = 0;

		int i = 0;
		int consecutiveCount = default;
		int tzcnt = default;
		while (n != 0)
		{
			tzcnt = math.tzcnt(n);
			cumulativeTzcnt += tzcnt;

			if (tzcnt == 0)
				consecutiveCount++;
			else
			{
				Debug.Log("Start:" + sequentialIndicesStartIndex);

				for (int j = 0; j < consecutiveCount; j++)
					dst[countSoFar + j] = src[sequentialIndicesStartIndex + j];
				countSoFar += consecutiveCount;

				consecutiveCount = 0;
				sequentialIndicesStartIndex = srcStartIndex + cumulativeTzcnt + i + 1;
			}

			//if (tzcnt == 0) // We check this because if it's zero that means we've hit a break and can't just copy this sequence of memory.
			//{
			//	consecutiveCount++;
			//}
			//else
			//{
			//	if(consecutiveCount > 0)
			//	{
			//		for (int j = 0; j < consecutiveCount; j++)
			//			dst[countSoFar + j] = srcPtr[sequentialIndicesStartIndex + j];
			//		countSoFar += consecutiveCount;
			//		consecutiveCount = 0;
			//		sequentialIndicesStartIndex = srcStartIndex + cumulativeTzcnt + i + 1;
			//	}
			//	else
			//	{
			//		dst[countSoFar] = src[srcStartIndex + cumulativeTzcnt + i];
			//		countSoFar++;
			//	}
			//}
			i++;

			n >>= tzcnt + 1;
		}

		//for (int j = 0; j < consecutiveCount; j++)
		//	dst[countSoFar + j] = srcPtr[sequentialIndicesStartIndex + j];
		countSoFar += consecutiveCount;

		counter.Value = countSoFar;
		//conditionalCopyMergeJobMarker.End();
	}

	public static JobHandle Schedule<M>(ConditionIndexingJob<T, M> indexingJob,
		NativeArray<T> dst,
		NativeReference<int> counter,
		JobHandle dependsOn = default) where M : IValidator<T>
	{
		ConditionalCopyMergeJob<T> job = new ConditionalCopyMergeJob<T>(indexingJob.src, dst, indexingJob.indices, counter);
		var handle = job.Schedule(indexingJob.indices.Length, dependsOn);

		return handle;
	}
}

#endregion

// This parallel setup is a fair bit faster, but also consumes far more of the CPU so probably not worth it in many cases
#region Parallel indexing, Single Sum, parallel copy

/// <summary>
/// This job is meant to pack booleans into <see cref="indices"/>.
/// Then you can use <see cref="ConditionalCopyMergeJob{T}"/> to write to a destination array based on the <see cref="indices"/> array.
/// </summary>
/// <typeparam name="T"></typeparam>
[BurstCompile, BurstCompatible]
public struct ConditionIndexingSumJob<T, M> : IJobParallelFor where T : unmanaged where M : IValidator<T>
{
	[ReadOnly]
	public M del;
	[ReadOnly]
	public NativeArray<T> src;
	[WriteOnly]
	public NativeArray<BitField64> indices;
	[WriteOnly]
	public NativeArray<int> counts;
	private static readonly ProfilerMarker conditionIndexingSumJobMarker = new ProfilerMarker(nameof(ConditionIndexingSumJob<T, M>));

	public ConditionIndexingSumJob(NativeArray<T> src, NativeArray<BitField64> indices, NativeArray<int> counts, M del = default)
	{
		this.src = src;
		this.indices = indices;
		this.counts = counts;
		this.del = del;
	}

	public void Execute(int index)
	{
		//conditionIndexingSumJobMarker.Begin();

		BitField64 bits = new BitField64(0);
		int dataIndex = index * 64;

		for (int i = 0; i < 64; i++)
			bits.SetBits(i, del.Validate(src[dataIndex + i]));

		counts[index] = math.countbits(bits.Value);
		indices[index] = bits;

		//conditionIndexingSumJobMarker.End();
	}

	private static readonly ProfilerMarker remainderJobMarker = new ProfilerMarker(nameof(RemainderSumJob));
	/// <summary>
	/// This job sets the bits and sums the last element of <see cref="indices"/>.
	/// It also will count all the bits at the end and store the count so far in <see cref="counts"/>.
	/// </summary>
	[BurstCompile, BurstCompatible]
	private struct RemainderSumJob : IJob
	{
		[ReadOnly]
		public M del;
		[ReadOnly]
		public NativeArray<T> src;
		[WriteOnly]
		public NativeArray<BitField64> indices;
		public NativeArray<int> counts;
		public NativeReference<int> totalCount;
		private BitField64 bits;

		public void Execute()
		{
			//remainderJobMarker.Begin();

			int remainderCount = src.Length % 64;
			int dataStartIndex = src.Length - remainderCount;
			bits.Clear();

			for (int i = 0; i < remainderCount; i++)
				bits.SetBits(i, del.Validate(src[dataStartIndex + i]));

			counts[indices.Length - 1] = math.countbits(bits.Value);
			indices[indices.Length - 1] = bits;

			// Lastly we want to count all of them together 
			for (int i = 0; i < counts.Length; i++)
			{
				totalCount.Value += counts[i];
				// We store the count so far because we can use it later
				counts[i] = totalCount.Value;
			}

			//remainderJobMarker.End();
		}
	}

	public static JobHandle Schedule(
		NativeArray<T> src,
		NativeArray<BitField64> indices,
		NativeArray<int> counts,
		NativeReference<int> totalCount,
		out ConditionIndexingSumJob<T, M> job,
		int innerBatchCount = 10,
		M del = default)
	{
		int remainder = src.Length % 64;
		// The job only supports writing whole 64 bit batches, so we floor here and then run the remainder elsewhere
		int length = (int)math.floor(src.Length / 64f);
		job = new ConditionIndexingSumJob<T, M>(src, indices, counts);

		var handle = job.Schedule(length, innerBatchCount);
		if (remainder > 0)
			handle = new RemainderSumJob()
			{
				src = src,
				indices = indices,
				counts = counts,
				totalCount = totalCount,
				del = del
			}.Schedule(handle);

		return handle;
	}
}

[BurstCompile, BurstCompatible]
public unsafe struct ParallelCopyJob<T> : IJobParallelFor where T : unmanaged
{
	[ReadOnly, NativeDisableParallelForRestriction]
	public NativeArray<T> src;
	[WriteOnly, NativeDisableParallelForRestriction]
	public NativeArray<T> dst;
	[ReadOnly]
	public NativeArray<int> counts;

	[NativeDisableUnsafePtrRestriction, ReadOnly]
	private readonly BitField64* bitsPtr;
	private readonly int indicesLength;
	private static readonly ProfilerMarker parallelCopyJobMarker = new ProfilerMarker(nameof(ParallelCopyJob<T>));

	public unsafe ParallelCopyJob(
		NativeArray<T> src,
		NativeArray<T> dst,
		NativeArray<BitField64> indices,
		NativeArray<int> counts)
	{
		this.src = src;
		this.dst = dst;
		indicesLength = indices.Length;
		this.counts = counts;
		bitsPtr = (BitField64*)indices.GetUnsafeReadOnlyPtr();
	}

	public void Execute(int index)
	{
		//parallelCopyJobMarker.Begin();

		// We need to start write index of the src data which we can get from counts
		int dstStartIndex = index == 0 ? 0 : counts[math.max(0, index - 1)];

		int length = src.Length;
		int srcStartIndex = index * 64;
		ulong n = bitsPtr[index].Value;

		int i = 0;
		int t = 0;
		while (n != 0)
		{
			int tcnt = math.tzcnt(n);
			t += tcnt;
			//Debug.Log($"Writing from index {srcStartIndex + t + i}");
			dst[dstStartIndex + i] = src[srcStartIndex + t + i];
			i++;
			n >>= tcnt + 1;
		}

		//parallelCopyJobMarker.End();
	}

	public static JobHandle Schedule<M>(ConditionIndexingSumJob<T, M> indexingJob,
		NativeArray<T> dst,
		int innerBatchCount = 10,
		JobHandle dependsOn = default) where M : IValidator<T>
	{
		ParallelCopyJob<T> job = new ParallelCopyJob<T>(indexingJob.src, dst, indexingJob.indices, indexingJob.counts);
		var handle = job.Schedule(indexingJob.indices.Length, innerBatchCount, dependsOn);

		return handle;
	}
}

#endregion