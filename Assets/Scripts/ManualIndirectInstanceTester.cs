using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
using System.Collections.Generic;
using Random = UnityEngine.Random;
using System.Runtime.InteropServices;

// Sort Input matrix buffer by a priority based on how often the developer thinks they will be updated.
// e.g if 10 objects will update often, put them in index 0 - 10.
// Cache all updates for any given frame in a index list, loop to find the min/max indexes to be updated,
// then create a subset to update the buffer.
// This could also we done by a job.

public class ManualIndirectInstanceTester : MonoBehaviour
{
	[System.Serializable, StructLayout(LayoutKind.Sequential)]
	struct CustomMatrix
	{
		public float4x4 mat;

		public CustomMatrix(float3x4 matrix)
		{
			this.mat = Matrix3x4To4x4(matrix);
		}
		public static implicit operator float4x4(CustomMatrix d) => d.mat;
		public static explicit operator CustomMatrix(float3x4 b) => new CustomMatrix(b);
	}

	[SerializeField]
	private Mesh mesh;
	[SerializeField]
	private int dimension = 10;
	[SerializeField]
	private float space = 1.5f;
	[SerializeField]
	private bool updateAllMatrices, updateSingleMatrix, updateMatrixSubset;
	[SerializeField, SetProperty(nameof(matrixSubsetUpdateMarker))]
	private int matrixUpdateSubsetCount = 10;
	[SerializeField, ReadOnly]
	private int dispatchX;

	public int OutputBufferCount => (dataBuffer != null) ? (dataBuffer.Count - 1) : 0;
	private int InputCount => dataBuffer.Count;
	private ComputeBuffer InputBuffer => dataBuffer.Buffer;
	private ComputeBuffer outputBuffer;
	private ComputeBuffer OutputBuffer
	{
		get => outputBuffer;
		set => outputBuffer = value;
	}
	public int TotalCount => dimension * dimension * dimension;
	private int MatrixUpdateSubsetCount
	{
		get => matrixUpdateSubsetCount;
		set
		{
			matrixUpdateSubsetCount = value;
			matrixSubsetUpdateMarker = $"Update Matrix Subset {matrixUpdateSubsetCount}";
		}
	}
	private string matrixSubsetUpdateMarker = "Update Matrix Subset";

	private Material mat;
	private Bounds bounds = new Bounds(Vector3.zero, Vector3.one * 10000);
	private uint[] args;
	private ComputeBuffer argsBuffer;
	private ShadowCastingMode shadowCastingMode = ShadowCastingMode.On;
	private GenericComputeBuffer<float3x4> dataBuffer;
	private GenericComputeBuffer<float4> colorBuffer;
	private ComputeShader appendCompute;
	private int appendComputeKernel = -1;
	private int threadCount = -1;
	private Transform camTrans;
	private List<float3x4> matrixBufferData = new List<float3x4>();
	private List<float4> colorBufferData = new List<float4>();

	protected static int computeInputID = Shader.PropertyToID("Input");
	protected static int computeOutputID = Shader.PropertyToID("Output");
	protected static int lengthID = Shader.PropertyToID("_Length");
	protected static int cameraPosID = Shader.PropertyToID("_CameraPos");
	protected static int materialMatrixBufferID = Shader.PropertyToID("matrixBuffer");
	protected static int materialIndexBufferID = Shader.PropertyToID("indexBuffer");
	protected static int colorBufferID = Shader.PropertyToID("colorBuffer");

	void Start()
	{
		Shader shader = Shader.Find("Unlit/InstancedIndirectUnlit");
		appendCompute = Resources.Load<ComputeShader>("InstancingAppendCompute");
		mat = new Material(shader);
		appendComputeKernel = appendCompute.FindKernel("CSMain");
		appendCompute.GetKernelThreadGroupSizes(appendComputeKernel, out uint x, out _, out _);
		threadCount = (int)x;
		camTrans = Camera.main.transform;

		dataBuffer = new GenericComputeBuffer<float3x4>(matrixBufferData);
		colorBuffer = new GenericComputeBuffer<float4>(colorBufferData);
		CreateTestMatricesAndPushToBuffer();
		CreateColorsAndPushToBuffer();

		OutputBuffer = new ComputeBuffer(OutputBufferCount, sizeof(uint), ComputeBufferType.Append);
		OutputBuffer.SetCounterValue(1);

		mat.SetBuffer(materialMatrixBufferID, InputBuffer);
		mat.SetBuffer(materialIndexBufferID, OutputBuffer);
		mat.SetBuffer(colorBufferID, colorBuffer.Buffer);

		InvalidateArgumentBuffer();
	}

	void Update()
	{
		// Performance tests for pushing data to GPU
		if (matrixBufferData.Count > 0)
		{
			if (updateAllMatrices)
				CreateTestMatricesAndPushToBuffer();
			else if (updateSingleMatrix)
			{
				int index = matrixBufferData.Count / 2;
				Vector3 middlePos = index * space * Vector3.one;
				matrixBufferData[index] = GetRandomMatrix(middlePos.x, middlePos.y, middlePos.z, space);

				Profiler.BeginSample("Update single matrix");
				dataBuffer.SetData(new DataSubset(index, 1));
				Profiler.EndSample();
			}
			else if (updateMatrixSubset)
			{
				int startIndex = Random.Range(matrixUpdateSubsetCount, matrixBufferData.Count - matrixUpdateSubsetCount - 1);
				int c = startIndex + matrixUpdateSubsetCount;
				for (int i = startIndex; i < c; i++)
				{
					Vector3Int index = new Vector3Int(Mathf.FloorToInt(i / dimension), i % dimension, 0);
					Vector3 pos = GetRandomSpacedVectorFromRoot(index.x, index.y, index.z, space);
					matrixBufferData[i] = GetRandomMatrix(pos.x, pos.y, pos.z, space);
				}

				Profiler.BeginSample(matrixSubsetUpdateMarker);
				dataBuffer.SetData(new DataSubset(0, matrixUpdateSubsetCount));
				Profiler.EndSample();
			}
		}

		Render();
	}
	//private uint Get3DimIndex(uint singleDimIndex, uint count)
	//{
	//	return i * count * count + j * count + k;
	//}
	private void InvalidateArgumentBuffer()
	{
		args = new uint[]
		{
			mesh.GetIndexCount(0),
			(uint)(dimension*dimension*dimension),
			mesh.GetIndexStart(0),
			mesh.GetBaseVertex(0),
			0
		};
		argsBuffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
		argsBuffer.SetData(args);
	}

	private void UpdateAppendCountInArgs()
	{
		//Graphics.CreateAsyncGraphicsFence
		// Copy count about 
		ComputeBuffer.CopyCount(OutputBuffer, argsBuffer, 4);
	}

	public void Render()
	{
		DispatchCompute();
		UpdateAppendCountInArgs();
		Graphics.DrawMeshInstancedIndirect(
				mesh,
				submeshIndex: 0,
				mat,
				bounds,
				argsBuffer,
				argsOffset: 0,
				properties: null,
				shadowCastingMode,
				receiveShadows: true,
				layer: gameObject.layer,
				camera: null,
				LightProbeUsage.BlendProbes,
				lightProbeProxyVolume: null);
	}

	private void DispatchCompute()
	{
		Profiler.BeginSample(nameof(DispatchCompute));

		OutputBuffer.SetCounterValue(0);
		appendCompute.SetBuffer(appendComputeKernel, computeInputID, dataBuffer.Buffer);
		appendCompute.SetBuffer(appendComputeKernel, computeOutputID, OutputBuffer);
		appendCompute.SetInt(lengthID, InputCount);
		appendCompute.SetVector(cameraPosID, camTrans.position);

		dispatchX = Mathf.Min((InputCount + threadCount - 1) / threadCount, 65535);
		appendCompute.Dispatch(appendComputeKernel, dispatchX, 1, 1);

		Profiler.EndSample();
	}

	private struct TempDataHolder
	{
		public float3x4 matrix;

		public TempDataHolder(float3x4 matrix)
		{
			this.matrix = matrix;
		}
	}

	private void CreateColorsAndPushToBuffer()
	{
		Profiler.BeginSample(nameof(CreateColorsAndPushToBuffer));
		colorBufferData.Clear();
		for (int i = 0; i < TotalCount; i++)
			colorBufferData.Add(GetRandomColor());
		Profiler.EndSample();

		var subset = new DataSubset(0, colorBufferData.Count - 1);

		Profiler.BeginSample("Update Colors and Set ComputeBuffer");
		colorBuffer.SetData(subset);
		Profiler.EndSample();
	}

	private TempDataHolder[,,] tempData;
	private void CreateTestMatricesAndPushToBuffer()
	{
		Profiler.BeginSample(nameof(CreateTestMatricesAndPushToBuffer));
		if (tempData == null ||
			tempData.GetLength(0) != dimension ||
			tempData.GetLength(1) != dimension ||
			tempData.GetLength(2) != dimension)
			tempData = new TempDataHolder[dimension, dimension, dimension];

		//float3x4 localMatrix = GetTransformMatrix(transform);
		float4x4 localMatrix = transform.worldToLocalMatrix;

		var testMatrix = GetTransformMatrixNoScale(new float3(1, 2, 3), float3x3.EulerXYZ(10, 20, 30));
		var multipliedMatrix = math.mul(testMatrix, localMatrix);

		for (int i = 0; i < dimension; i++)
		{
			for (int j = 0; j < dimension; j++)
			{
				for (int k = 0; k < dimension; k++)
				{
					//tempMatrices[i, j, k] = GetRandomMatrix(i, j, k, space);
					tempData[i, j, k] = new TempDataHolder(
						math.mul(GetRandomMatrix(i,j,k, space), localMatrix));
				}
			}
		}
		Profiler.EndSample();		

		Profiler.BeginSample("Update Data list");
		matrixBufferData.Clear();
		colorBufferData.Clear();
		for (int i = 0; i < dimension; i++)
		{
			for (int j = 0; j < dimension; j++)
			{
				for (int k = 0; k < dimension; k++)
				{
					matrixBufferData.Add(tempData[i, j, k].matrix);
				}
			}
		}
		Profiler.EndSample();
		
		var subset = new DataSubset(0, matrixBufferData.Count - 1);

		Profiler.BeginSample("Update Matrices and Set ComputeBuffer");
		dataBuffer.SetData(subset);
		Profiler.EndSample();
	}

	private static float3x4 GetTransformMatrix(Transform transform)
	{
		float4x4 matrix = transform.worldToLocalMatrix;
		return new float3x4(Float4To3(matrix.c0), Float4To3(matrix.c1), Float4To3(matrix.c2), Float4To3(matrix.c3));
	}
	private static float3x4 GetTransformMatrixNoScale(float3 position, float3x3 rotationMatrix)
	{
		// float3x4 = Row first, column second
		// Unity is column major
		return math.float3x4(rotationMatrix.c0, rotationMatrix.c1, rotationMatrix.c2, position);
	}

	private static float3x4 GetRandomMatrix(float rootX, float rootY, float rootZ, float space) =>
		GetTransformMatrixNoScale(GetRandomSpacedVectorFromRoot(rootX, rootY, rootZ, space), GetRotationMatrix(GetRandomRotation()));
	private static float4 GetRandomColor() => new float4(Random.value, Random.value, Random.value, 1);
	private static Vector3 GetRandomSpacedVectorFromRoot(float x, float y, float z, float space) =>
		new Vector3(x * space, y * space, z * space) + GetRandomVector();
	private static Vector3 GetRandomVector() => new Vector3(Random.value, Random.value, Random.value);
	private static Quaternion GetRandomRotation() => Quaternion.Euler(Random.value * 360, Random.value * 360, Random.value * 360);
	private static float3x3 GetRotationMatrix(Quaternion quaternion) => float3x3.EulerXYZ(quaternion.eulerAngles);
	private static float4x4 Matrix3x4To4x4(float3x4 matrix) =>
		new float4x4(Float3To4(matrix.c0), Float3To4(matrix.c1), Float3To4(matrix.c2), Float3To4(matrix.c3));
	private static float3 Float4To3(float4 vector) => new float3(vector.x, vector.y, vector.z);
	private static float4 Float3To4(float3 vector) => new float4(vector.x, vector.y, vector.z, 1);
}