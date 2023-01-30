using CommonJobs;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using static MathUtil;

public interface IValidator<T> where T : unmanaged
{
	bool Validate(T element);
}

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

	public ConditionIndexingJob(NativeArray<T> src, NativeArray<BitField64> indices, M del = default)
	{
		this.src = src;
		this.indices = indices;
		this.del = del;
	}

	public void Execute(int index)
	{
		BitField64 bits = new BitField64(0);
		//byte* bitsPtr = (byte*)&bits.Value;
		//const int iterations = 64 / 4;
		int dataIndex = index * 64;

		for (int i = 0; i < 64; i++)
			bits.SetBits(i, del.Validate(src[dataIndex + i]));
		indices[index] = bits;
	}

	[BurstCompile, BurstCompatible]
	private struct RemainderJob : IJob
	{
		[ReadOnly]
		public M del;
		[ReadOnly]
		public NativeArray<T> src;
		[WriteOnly]
		public NativeArray<BitField64> indices;

		public void Execute()
		{
			int remainderCount = src.Length % 64;
			int dataStartIndex = src.Length - remainderCount;
			BitField64 bits = new BitField64(0);

			for (int i = 0; i < remainderCount; i++)
			{
				//ulong mask = 1u << i;
				//if (del.Validate(src[dataStartIndex + i]))
				//	bits.Value |= mask;
				//else
				//	bits.Value &= ~mask;
				bits.SetBits(i, del.Validate(src[dataStartIndex + i]));
			}

			indices[indices.Length - 1] = bits;
		}
	}

	public static JobHandle Schedule(NativeArray<T> src, NativeArray<BitField64> indices, out ConditionIndexingJob<T, M> job, M del = default)
	{
		int remainder = src.Length % 64;
		// The job only supports writing whole 64 bit batches, so we floor here and then run the remainder elsewhere
		int length = (int)math.floor(src.Length / 64f);
		job = new ConditionIndexingJob<T, M>(src, indices);

		var handle = job.Schedule(length, default);
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
public unsafe struct ConditionalCopyMergeJob<T> : IJob where T : unmanaged
{
	[ReadOnly]
	public NativeArray<T> src;
	[/*WriteOnly,*/ NativeDisableParallelForRestriction]
	public NativeArray<T> dst;

	[NativeDisableUnsafePtrRestriction, ReadOnly]
	private readonly BitField64* bitsPtr;
	[NativeDisableUnsafePtrRestriction, ReadOnly]
	private readonly int* indexPtr;
	private readonly int indicesLength;

	[WriteOnly]
	private NativeReference<int> counter;

	public unsafe ConditionalCopyMergeJob(
		NativeArray<T> src,
		NativeArray<T> dst,
		NativeArray<BitField64> indices,
		NativeReference<int> count,
		NativeArray<int> indexArray)
	{
		this.src = src;
		this.dst = dst;
		indicesLength = indices.Length;
		this.counter = count;
		bitsPtr = (BitField64*)indices.GetUnsafeReadOnlyPtr();
		indexPtr = (int*)indexArray.GetUnsafeReadOnlyPtr();
	}

	public void Execute(/*int index*/)
	{
		int count = 0;
		int length = src.Length;
		for (int index = 0; index < indicesLength; index++)
		{
			BitField64* mask = &bitsPtr[index];

			int srcStartIndex = index * 64;
			ulong n = mask->Value;
			
			int i = 0;
			int t = 0;
			while (n != 0)
			{
				int tcnt = math.tzcnt(n);
				t += tcnt;
				//Debug.Log($"Writing from index {srcStartIndex + t + i}");
				dst[count] = src[srcStartIndex + t + i];
				i++;
				count++;
				n >>= tcnt + 1;
			}
		}
		counter.Value = count;
	}

	public static JobHandle Schedule<M>(ConditionIndexingJob<T, M> indexingJob,
		NativeArray<T> dst,
		NativeReference<int> counter,
		NativeArray<int> indexArray = default, JobHandle dependsOn = default) where M : IValidator<T>
	{
		bool hasIndexArray = indexArray.IsCreated;
		if (!hasIndexArray)
			indexArray = new NativeArray<int>(64, Allocator.TempJob);

		ConditionalCopyMergeJob<T> job = new ConditionalCopyMergeJob<T>(indexingJob.src, dst, indexingJob.indices, counter, indexArray);
		var handle = job.Schedule(dependsOn);

		if (!hasIndexArray)
			handle = indexArray.Dispose(handle);

		return handle;
	}
}