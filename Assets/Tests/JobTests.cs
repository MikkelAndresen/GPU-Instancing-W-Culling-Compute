using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using Unity.Jobs;
using Random = Unity.Mathematics.Random;
using System;
using Unity.Collections.LowLevel.Unsafe;

public class JobTests
{	
	private struct GreaterThanZeroDel : IValidator<float>
	{
		public bool Validate(float element) => element > 0;
	}

	[Test]
	public unsafe void ParallelBitIndexingJob()
	{
		int halfLength = 50;
		NativeArray<float> data = new NativeArray<float>(halfLength * 2, Allocator.Persistent);
		for (int i = 0; i < data.Length; i++)
			data[i] = i % 2 == 0 ? -1 : i;

		NativeArray<BitField64> bits = new NativeArray<BitField64>((int)math.ceil(100f / 64f), Allocator.Persistent);
		ConditionIndexingJob<float, GreaterThanZeroDel>.Schedule(data, bits, out var job).Complete();
		//Debug.Log(Convert.ToString((long)bits[0].Value, toBase: 2));

		int bitCount = 0;
		for (int i = 0; i < bits.Length; i++)
			bitCount += math.countbits(bits[i].Value);
		Assert.AreEqual(halfLength, bitCount);

		NativeArray<float> dstData = new NativeArray<float>(100, Allocator.Persistent);
		NativeReference<int> counter = new NativeReference<int>(0, Allocator.Persistent);

		ConditionalCopyMergeJob<float>.Schedule(job, dstData, counter).Complete();

		int count = counter.Value;
		counter.Dispose();
		Assert.AreEqual(bitCount, count);

		bits.Dispose();

		// We copy all the data we wish to assert because if an assertion fails
		// we get exceptions due to native collections not being disposed.
		float[] dataCopy = new float[data.Length];
		data.CopyTo(dataCopy);
		float[] dstCopy = new float[dstData.Length];
		dstData.CopyTo(dstCopy);

		data.Dispose();
		dstData.Dispose();

		Assert.AreEqual(dstCopy.Length, dataCopy.Length);
		var comparer = new GreaterThanZeroDel();
		for (int i = 0; i < count && comparer.Validate(dataCopy[i]); i++)
		{
			//Debug.Log($"Result: {dataCopy[i]}");

			if (comparer.Validate(dataCopy[i]))
				Assert.AreEqual(dataCopy[i], dstCopy[i], $"Index {i} had the wrong value after validation");
			else
				Assert.AreNotEqual(-1, dstCopy[i], $"Index {i} had the wrong value after correct false validation");
		}
	}
}