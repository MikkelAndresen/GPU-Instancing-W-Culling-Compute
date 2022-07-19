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

[BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
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

[BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
public struct GenerateMatricesJob : IJobParallelFor
{
	[ReadOnly]
	private float3 pos;
	[ReadOnly]
	private quaternion rot;
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
		float3 pos,
		quaternion rot,
		float space,
		int dimension,
		float theta)
	{
		this.matrices = matrices;
		this.space = space;
		this.dimension = dimension;
		this.theta = theta;
		this.pos = pos;
		this.rot = rot;
	}

	public void Execute(int i) => UpdateMatrix(ref matrices, pos, rot, i, dimension, space, theta);

	public static void UpdateMatrix(ref NativeArray<float3x4> matrices, float3 pos, quaternion rot, int i, int dimension, float space, float theta)
	{
		int3 index3D = MathUtil.Get3DIndex(i, dimension, dimension);
		var mat3x4 = TestDataGenerator.GetTransformMatrixNoScale(
			new float3(index3D.x, index3D.y, index3D.z) * space,
			quaternion.EulerXYZ(theta, -theta, theta));

		var mat4x4 = new float4x4(new float4(mat3x4.c0, 0), new float4(mat3x4.c1, 0), new float4(mat3x4.c2, 0), new float4(mat3x4.c3, 1));
		float4x4 parentMat4 = new float4x4(rot, pos);
		mat4x4 = math.mul(parentMat4, mat4x4);

		matrices[i] = new float3x4(mat4x4.c0.xyz, mat4x4.c1.xyz, mat4x4.c2.xyz, mat4x4.c3.xyz);

		//matrices[i] = TestDataGenerator.GetTransformMatrixNoScale(
		//	pos + (new float3(index3D.x, index3D.y, index3D.z) * space),
		//	math.mul(rot, quaternion.EulerXYZ(theta, -theta, theta)));
		//matrices[i] = TestDataGenerator.GetTransformMatrixNoScale(new float3(2, 2, 2), quaternion.identity);
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
	private readonly Stack<(JobHandle, Action)> callbackStack = new Stack<(JobHandle, Action)>();
	private readonly Transform parent;

	private float theta;
	public float Theta
	{
		get => theta;
		set => theta = math.min(value, 360);
	}

	public TestDataGenerator(int dimension, Transform parent = null)
	{
		this.dimension = dimension;
		random = new Random();
		random.InitState();
		EnsureArraySize(ref matrices, Count);
		EnsureArraySize(ref colors, Count);
		this.parent = parent;
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

	private readonly ProfilerMarker runMatrixJobMarker = new ProfilerMarker(nameof(RunMatrixJob));
	public JobHandle RunMatrixJob(ref NativeArray<float3x4> matrices, float space, bool completeNow, float deltaTime = -1)
	{
		runMatrixJobMarker.Begin();

		deltaTime = completeNow ? Time.deltaTime : deltaTime;
		Theta += deltaTime * anglePerSecond;

		quaternion parentRot = parent != null ? (quaternion)parent.rotation : quaternion.identity;
		float3 parentPos = parent != null ? (float3)parent.position : float3.zero;
		GenerateMatricesJob job = new GenerateMatricesJob(matrices, parentPos, parentRot, space, dimension, theta);

		try // We trycatch here because other jobs might be using the array
		{
			currentMatrixJob = job.Schedule(matrices.Length, 1);
		}
		catch (Exception e)
		{
			Debug.LogException(e);
		}
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