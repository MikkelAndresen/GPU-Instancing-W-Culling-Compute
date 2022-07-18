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
			float side = math.dot(point, frustumPlanes[i].normal) + frustumPlanes[i].distance;
			if (side < 0.0f)
				return false;
		}

		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsSphereInFrustum(NativeArray<Plane> frustumPlanes, float3 center, float radius)
	{
		for (int i = 0; i < 6; i++)
		{
			float side = math.dot(center, frustumPlanes[i].normal) + frustumPlanes[i].distance;
			if (side < -radius)
				return false;
		}

		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsBoundsInFrustum(NativeArray<Plane> frustumPlanes, AABB bounds)
	{
		for (int i = 0; i < 6; i++)
		{
			if (bounds.ClassifyAgainstPlane(frustumPlanes[i]) < 0)
				return false;
		}

		return true;
	}

	[System.Serializable, BurstCompatible, StructLayout(LayoutKind.Sequential), BurstCompatible]
	public struct AABB
	{
		public AABB(float3 center, float3 size)
		{
			Center = center;
			Size = size;
		}

		/// <summary>
		/// Returns 0 if this bounds intersects the plane.
		/// Returns a positive number if this bounds is in front of the plane or a negative when behind.
		/// </summary>
		/// <param name="plane"></param>
		/// <returns></returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public float ClassifyAgainstPlane(Plane plane)
		{
			float r = math.abs(Size.x * plane.normal.x)
					 + math.abs(Size.y * plane.normal.y)
					 + math.abs(Size.z * plane.normal.z);
			float d = math.dot(plane.normal, Center)
			 + plane.distance;
			if (math.abs(d) < r)
				return 0f;
			else if (d < 0.0f)
				return d + r;
			return d - r;
		}

		public float3 Center { get; private set; }
		public float3 Size { get; private set; }
	}

	[System.Serializable, BurstCompatible, StructLayout(LayoutKind.Sequential), BurstCompatible]
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
}