using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Mathematics;

public static class MathUtil
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int3 Get3DIndex(int index, int dimX, int dimY) => new int3(index % dimX, (index / dimX) % dimY, index / (dimY * dimX));

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsPointInFrustum(NativeArray<Plane> frustumPlanes, float3 point)
	{
		for (int i = 0; i < 6; i++)
		{
			// TODO Add more plane comparisons for AABB and spheres
			float side = math.dot(point, frustumPlanes[i].normal) + frustumPlanes[i].distance;
			if (side < 0.0f)
				return false;
		}

		return true;
	}

	[System.Serializable, StructLayout(LayoutKind.Sequential), BurstCompatible]
	public struct Plane
	{
		public float3 normal;
		public float distance;
		public Plane(UnityEngine.Plane uP)
		{
			this.normal = uP.normal;
			this.distance = uP.distance;
		}
		public Plane(float3 normal, float distance)
		{
			this.normal = normal;
			this.distance = distance;
		}
	}

	public struct Sphere
	{
		public float3 point;
		public float radius;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsSphereInFrustum(NativeArray<Plane> frustumPlanes, Sphere sphere)
	{
		for (int i = 0; i < 6; i++)
		{
			float side = math.dot(sphere.point, frustumPlanes[i].normal) + frustumPlanes[i].distance;
			if (side < -sphere.radius)
				return false;
		}
		return true;
	}
}