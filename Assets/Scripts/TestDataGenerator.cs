using System;
using Unity.Jobs;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Profiling;
using Random = Unity.Mathematics.Random;
using Debug = UnityEngine.Debug;
using Unity.Burst;
using System.Runtime.CompilerServices;
using System.Collections.Generic;
using Unity.Profiling;

[BurstCompile]
public struct GenerateColorsJob : IJobParallelFor
{
	[ReadOnly]
	private Random r;
	[WriteOnly]
	private NativeArray<float4> colors;

	public GenerateColorsJob(Random r, NativeArray<float4> colors)
	{
		this.r = r;
		this.colors = colors;
	}

	public void Execute(int index) => colors[index] = r.NextFloat4();
}

[BurstCompile]
public struct GenerateMatricesJob : IJobParallelFor
{
	[WriteOnly]
	private NativeArray<float3x4> matrices;
	[ReadOnly]
	private float space;
	[ReadOnly]
	private int dimension;
	[ReadOnly]
	private float theta;

	public GenerateMatricesJob(
		NativeArray<float3x4> matrices,
		float space,
		int dimension,
		float theta)
	{
		this.matrices = matrices;
		this.space = space;
		this.dimension = dimension;
		this.theta = theta;
	}

	public void Execute(int i) => UpdateMatrix(ref matrices, i, dimension, space, theta);
	
	public static void UpdateMatrix(ref NativeArray<float3x4> matrices, int i, int dimension, float space, float theta)
	{
		int3 index3D = MathUtil.Get3DIndex(i, dimension, dimension);
		matrices[i] = TestDataGenerator.GetTransformMatrixNoScale(
			new float3(index3D.x, index3D.y, index3D.z) * space,
			quaternion.EulerXYZ(theta, -theta, theta));
	}
}

public class TestDataGenerator : IDisposable
{
	public int dimension;
	public int Count => dimension * dimension * dimension;
	public NativeArray<float3x4> matrices;
	public NativeArray<float4> colors;
	public float anglePerSecond = 1;

	private Random random;
	private JobHandle currentMatrixJob;
	private Stack<(JobHandle, Action)> callbackStack = new Stack<(JobHandle, Action)>();

	private float theta;
	public float Theta
	{
		get => theta;
		set => theta = math.min(value, 360);
	}

	public TestDataGenerator(int dimension)
	{
		this.dimension = dimension;
		random = new Random();
		random.InitState();
		EnsureArraySize(ref matrices, Count);
		EnsureArraySize(ref colors, Count);
	}

	private JobHandle currentColorJob;
	public void RunColorJob(int dimension, bool completeNow)
	{
		Profiler.BeginSample(nameof(RunColorJob));
		if (currentColorJob.IsCompleted)
		{
			currentColorJob.Complete();

			int count = dimension * dimension * dimension;
			if (dimension != this.dimension)
			{
				this.dimension = dimension;
				EnsureArraySize(ref colors, count);
			}

			GenerateColorsJob job = new GenerateColorsJob(random, colors);
			currentColorJob = job.Schedule(colors.Length, 1);
			if (completeNow)
				currentColorJob.Complete();
		}
		Profiler.EndSample();
	}

	public JobHandle RunMatrixJob(int dimension, float space, bool completeNow, float deltaTime = -1)
	{
		int count = dimension * dimension * dimension;
		if (dimension != this.dimension)
		{
			this.dimension = dimension;
			EnsureArraySize(ref matrices, count);
		}
		return RunMatrixJob(ref matrices, space, completeNow, deltaTime);
	}

	private ProfilerMarker runMatrixJobMarker = new ProfilerMarker(nameof(RunMatrixJob));
	public JobHandle RunMatrixJob(ref NativeArray<float3x4> matrices, float space, bool completeNow, float deltaTime = -1)
	{
		runMatrixJobMarker.Begin();

		deltaTime = completeNow ? Time.deltaTime : deltaTime;
		Theta += deltaTime * anglePerSecond;

		GenerateMatricesJob job = new GenerateMatricesJob(matrices, space, dimension, theta);
		currentMatrixJob = job.Schedule(matrices.Length, 1);
		if (completeNow)
			currentMatrixJob.Complete();

		runMatrixJobMarker.End();
		return currentMatrixJob;
	}

	private void EnsureArraySize<T>(ref NativeArray<T> array, int newCount) where T : unmanaged
	{
		if (array != default)
		{
			if (array.Length == newCount)
				return;

			array.Dispose();
		}
		array = new NativeArray<T>(newCount, Allocator.Persistent);
	}

	public void Dispose()
	{
		currentMatrixJob.Complete();
		matrices.Dispose();
		colors.Dispose();
	}

	public static float3x4 GetTransformMatrix(Transform transform)
	{
		float4x4 matrix = transform.worldToLocalMatrix;
		return new float3x4(Float4To3(matrix.c0), Float4To3(matrix.c1), Float4To3(matrix.c2), Float4To3(matrix.c3));
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float3x4 GetTransformMatrixNoScale(float3 position, quaternion rotation)
	{
		float3x3 rotationMatrix = new float3x3(rotation);
		// float3x4 = Row first, column second
		// Unity is column major
		return math.float3x4(rotationMatrix.c0, rotationMatrix.c1, rotationMatrix.c2, position);
	}

	public static quaternion GetRandomQuaternion(ref Random r) => r.NextQuaternionRotation();
	public static float3x4 GetRandomMatrix(float rootX, float rootY, float rootZ, float space, ref Random r) =>
		GetTransformMatrixNoScale(GetRandomSpacedVectorFromRoot(rootX, rootY, rootZ, space, ref r), GetRandomQuaternion(ref r));
	public static float4 GetRandomColor(ref Random r) => new float4(r.NextFloat(), r.NextFloat(), r.NextFloat(), 1);
	public static Vector3 GetRandomSpacedVectorFromRoot(float x, float y, float z, float space, ref Random r) =>
		new float3(x * space, y * space, z * space) + r.NextFloat3();
	public static float4x4 Matrix3x4To4x4(float3x4 matrix) =>
		new float4x4(Float3To4(matrix.c0), Float3To4(matrix.c1), Float3To4(matrix.c2), Float3To4(matrix.c3));
	public static float3 Float4To3(float4 vector) => new float3(vector.x, vector.y, vector.z);
	public static float4 Float3To4(float3 vector) => new float4(vector.x, vector.y, vector.z, 1);
}