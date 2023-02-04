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

	private struct ValidateTrue : IValidator<float>
	{
		public bool Validate(float element) => true;
	}

	private struct ValidateFalse : IValidator<float>
	{
		public bool Validate(float element) => false;
	}

	[Test]
	public void TestCopyAll1Bits() => TestParallelIndexSingleCopy<ValidateTrue>((i) => i);
	
	[Test]
	public void TestCopyAll0Bits() => TestParallelIndexSingleCopy<ValidateFalse>((i) => i);

	[Test]
	public void TestCopyAllOddBits() => TestParallelIndexSingleCopy<GreaterThanZeroDel>((i) => i % 2 == 0 ? -1f : 1);

	[Test]
	public void TestCopyAllBatchedBits()
	{
		int j = 0;
		TestParallelIndexSingleCopy<GreaterThanZeroDel>((i) => 
		{
			j++;
			if (j >= 5)
				j = -5;
			return j > 0 ? 1 : -1;
		});
	}

	private static void TestParallelIndexSingleCopy<T>(Func<float, float> dataGen) where T : unmanaged, IValidator<float>
	{
		// We declare this as a method because we want to use it multiple times later
		//float GetTestValue(float i) => i;//i % 2 == 0 ? -1f : i; 

		NativeArray<float> src = new NativeArray<float>(100, Allocator.Persistent);
		for (int i = 0; i < src.Length; i++)
		{
			src[i] = dataGen(i);
			//Debug.Log("D:"+data[i]);
		}

		NativeArray<BitField64> bits = new NativeArray<BitField64>((int)math.ceil(100f / 64f), Allocator.Persistent);
		ConditionIndexingJob<float, T>.Schedule(src, bits, out var job).Complete();
		//Debug.Log(Convert.ToString((long)bits[0].Value, toBase: 2));

		NativeArray<float> dstData = new NativeArray<float>(100, Allocator.Persistent);
		NativeReference<int> counter = new NativeReference<int>(0, Allocator.Persistent);

		ConditionalCopyMergeJob<float>.Schedule(job, dstData, counter).Complete();

		int count = counter.Value;
		counter.Dispose();
		bits.Dispose();

		// We copy all the data we wish to assert because if an assertion fails
		// we get exceptions due to native collections not being disposed.
		float[] srcCopy = new float[src.Length];
		src.CopyTo(srcCopy);
		float[] dstCopy = new float[dstData.Length];
		dstData.CopyTo(dstCopy);

		src.Dispose();
		dstData.Dispose();
		(float[] expected, int expectedLength) = GetExpected<T>(srcCopy);

		Assert.AreEqual(expectedLength, count, "Incorrect length");
		Assert.AreEqual(dstCopy.Length, srcCopy.Length, "Differing length between dst and src");

		for (int i = 0; i < dstData.Length; i++)
		{
			//Debug.Log($"Expected/Actual: {expected[i]}/{dstCopy[i]}");
			Assert.AreEqual(expected[i], dstCopy[i], $"Index {i} had the wrong value");
		}
	}

	private static (float[] arr, int expectedLength) GetExpected<T>(float[] data) where T : IValidator<float>
	{
		T comparer = default;
		float[] expected = new float[100];
		int j = 0;
		for (int i = 0; i < expected.Length; i++)
		{
			if (comparer.Validate(data[i]))
			{
				expected[j] = data[i];
				j++;
			}
		}
		return (expected, j);
	}
}