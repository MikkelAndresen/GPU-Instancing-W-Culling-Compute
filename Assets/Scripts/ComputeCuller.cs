using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Profiling;
using UnityEngine;

public class ComputeCuller : IDisposable
{
	public readonly static int computeInputID = Shader.PropertyToID("Input");
	public readonly static int computeOutputID = Shader.PropertyToID("Output");
	public readonly static int lengthID = Shader.PropertyToID("_Length");
	public readonly static int maxDistanceID = Shader.PropertyToID("_MaxDistance");
	public readonly static int cameraPosID = Shader.PropertyToID("_CameraPos");
	public readonly static int frustumBufferID = Shader.PropertyToID("_FrustumPlanes");
	public readonly static int ComputeKernel = CullingCompute.FindKernel("CSMain");

	public readonly static uint threadCount;
	private static ComputeShader cullingCompute;
	public static ComputeShader CullingCompute => cullingCompute == null || cullingCompute.Equals(null) ?
		cullingCompute = Resources.Load<ComputeShader>("MatrixFrustumCullingCompute") : cullingCompute;
	static ComputeCuller() 
	{
		CullingCompute.GetKernelThreadGroupSizes(ComputeKernel, out uint x, out _, out _);
		threadCount = x;
	}

	private readonly ComputeBuffer frustumPlaneBuffer;
	private readonly Plane[] frustumPlanes = new Plane[6];
	private NativeArray<Plane> nativeFrustumPlanes = new NativeArray<Plane>(6, Allocator.Persistent);
	public NativeArray<Plane> FrustumPlanes => nativeFrustumPlanes;

	public Camera Cam { get; private set; }
	public ComputeCuller(Camera cam)
	{
		this.Cam = cam != null ? cam : throw new ArgumentNullException(nameof(cam));
		frustumPlaneBuffer = new ComputeBuffer(6, Marshal.SizeOf(typeof(Plane)));
	}

	private readonly ProfilerMarker invalidateFrustumPlanesMarker = new ProfilerMarker("Invalidate Frustum planes cache");
	private void InvalidateFrustumPlaneCache()
	{
		invalidateFrustumPlanesMarker.Begin();

		GeometryUtility.CalculateFrustumPlanes(Cam, frustumPlanes);
		frustumPlaneBuffer.SetData(frustumPlanes);
		for (int i = 0; i < frustumPlanes.Length; i++)
			nativeFrustumPlanes[i] = frustumPlanes[i];

		invalidateFrustumPlanesMarker.End();
	}

	private readonly ProfilerMarker dispatchComputeMarker = new ProfilerMarker("Dispatch Culling Compute");
	public void DispatchCompute(ComputeBuffer matrixBuffer, ComputeBuffer indexBuffer)
	{
		if (matrixBuffer == null)
			throw new ArgumentNullException(nameof(matrixBuffer));
		if (indexBuffer == null)
			throw new ArgumentNullException(nameof(indexBuffer));

		int count = matrixBuffer.count;
		if (matrixBuffer.count != indexBuffer.count)
		{
			Debug.LogWarning($"{nameof(ComputeCuller)} received matrix and index buffer with differing counts");
			return;
		}
		InvalidateFrustumPlaneCache();

		dispatchComputeMarker.Begin();

		CullingCompute.SetBuffer(ComputeKernel, computeInputID, matrixBuffer); // Matrices
		CullingCompute.SetBuffer(ComputeKernel, computeOutputID, indexBuffer); // Indices
		CullingCompute.SetBuffer(ComputeKernel, frustumBufferID, frustumPlaneBuffer); // Frustum planes

		CullingCompute.SetInt(lengthID, count);
		CullingCompute.SetVector(cameraPosID, Cam.transform.position);
		CullingCompute.SetFloat(maxDistanceID, 100f);
		int tc = (int)threadCount;
		int DispatchX = Mathf.Min((count + tc - 1) / tc, 65535);

		CullingCompute.Dispatch(ComputeKernel, DispatchX, 1, 1);

		dispatchComputeMarker.End();
	}

	public void Dispose()
	{
		frustumPlaneBuffer.Dispose();
		nativeFrustumPlanes.Dispose();
	}
}