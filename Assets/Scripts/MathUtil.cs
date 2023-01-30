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

	[System.Serializable, BurstCompatible, StructLayout(LayoutKind.Sequential), BurstCompatible]
	public struct AABB
	{
		public float3 Center { get; private set; }
		public float3 Size { get; private set; }
		public float3 Extents => Size / 2f;

		public AABB(in float3 center, in float3 size)
		{
			Center = center;
			Size = size;
		}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public bool IsBoundsInFrustum(NativeArray<Plane> frustumPlanes)
	{
		for (int i = 0; i < 6; i++)
		{
				if (ClassifyAgainstPlane(frustumPlanes[i]) < 0)
				return false;
		}

		return true;
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

		/// <summary>
		/// The minimum coordinate of the AABB
		/// </summary>
		/// <returns>The minimum coordinate of the AABB, in three dimensions.</returns>
		public float3 Min => Center - Extents;

		/// <summary>
		/// The maximum coordinate of the AABB
		/// </summary>
		/// <returns>The maximum coordinate of the AABB, in three dimensions.</returns>
		public float3 Max => Center + Extents;
	}

	/// <summary>
	/// This is meant for manual vectorization of functions such as raycasting
	/// </summary>
	[System.Serializable, BurstCompatible]
	public unsafe struct AABB4
	{
		public AABB x, y, z, w;
		/// <summary>
		/// The location of the center of the AABB
		/// </summary>
		public float3x4 Center => new float3x4(x.Center, y.Center, z.Center, w.Center);

		/// <summary>
		/// A 3D vector from the center of the AABB, to the corner of the AABB with maximum XYZ values
		/// </summary>
		public float3x4 Extents => new float3x4(x.Extents, y.Extents, z.Extents, w.Extents);

		public AABB4(in AABB x, in AABB y, in AABB z, in AABB w) : this()
		{
			this.x = x;
			this.y = y;
			this.z = z;
			this.w = w;
		}

		/// <summary>Returns the AABB element at a specified index.</summary>
		/// Taken from float4.gen.cs line 3240
		unsafe public AABB this[int index]
		{
			get
			{
#if ENABLE_UNITY_COLLECTIONS_CHECKS
				if ((uint)index >= 4)
					throw new System.ArgumentException("index must be between[0...3]");
#endif
				fixed (AABB4* array = &this) { return ((AABB*)array)[index]; }
			}
			set
			{
#if ENABLE_UNITY_COLLECTIONS_CHECKS
				if ((uint)index >= 4)
					throw new System.ArgumentException("index must be between[0...3]");
#endif
				fixed (AABB* array = &x) { array[index] = value; }
			}
		}

		/// <summary>
		/// The size of the AABB
		/// </summary>
		/// <returns>The size of the AABB, in three dimensions. All three dimensions must be positive.</returns>
		public float3x4 Size => Extents * 2;

		/// <summary>
		/// The minimum coordinate of the AABB
		/// </summary>
		/// <returns>The minimum coordinate of the AABB, in three dimensions.</returns>
		public float3x4 Min => Center - Extents;

		/// <summary>
		/// The maximum coordinate of the AABB
		/// </summary>
		/// <returns>The maximum coordinate of the AABB, in three dimensions.</returns>
		public float3x4 Max => Center + Extents;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public bool4 IsBoundsInFrustum(NativeArray<Plane> frustumPlanes)
		{
			float4x3 size = math.transpose(Size);
			float4x3 center = math.transpose(Center);

			bool4 result = true;
			for (int i = 0; i < 6; i++)
				result &= AABB4.ClassifyAgainstPlane(frustumPlanes[i], size, center) >= 0;
			return result;
		}

		/// <summary>
		/// Returns 0 if this bounds intersects the plane.
		/// Returns a positive number if this bounds is in front of the plane or a negative when behind.
		/// </summary>
		/// <param name="plane"></param>
		/// <returns></returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public float4 ClassifyAgainstPlane(in Plane plane)
		{
			float4x3 size = math.transpose(Size);
			float4x3 center = math.transpose(Center);

			return ClassifyAgainstPlane(plane, size, center);
		}

		/// <summary>
		/// Returns 0 if this bounds intersects the plane.
		/// Returns a positive number if this bounds is in front of the plane or a negative when behind.
		/// </summary>
		/// <param name="plane"></param>
		/// <returns></returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static float4 ClassifyAgainstPlane(in Plane plane, in float4x3 size, in float4x3 center)
		{
			float4 r = math.abs(size.c0 * plane.normal.x) +
						math.abs(size.c1 * plane.normal.y) +
						math.abs(size.c2 * plane.normal.z);

			float4 d = Dot4(center.c0, center.c1, center.c2, plane.normal)
			 + plane.distance;

			return math.select(math.select(d - r, d + r, d < 0.0f), 0f, math.abs(d) < r);
		}
	}

	[System.Serializable, BurstCompatible, StructLayout(LayoutKind.Sequential), BurstCompatible]
	public struct Plane
	{
		public float3 normal;
		public float distance;
		public Plane(UnityEngine.Plane up)
		{
			this.normal = up.normal;
			this.distance = up.distance;
		}
		public Plane(float3 normal, float distance)
		{
			this.normal = normal;
			this.distance = distance;
		}
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float4 Dot4(float4 x, float4 y, float4 z, float3 rhs) => x * rhs.x + y * rhs.y + z * rhs.z;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public unsafe static int GetSetBitIndices(ulong n, int* indices)
	{
		int i = 0;
		int t = 0;
		while (n != 0)
		{
			int tcnt = math.tzcnt(n);
			t += tcnt;
			indices[i] = t + i;
			//Debug.Log($"Found bit index: {t + i}");
			i++;

			n >>= tcnt + 1;
		}
		return i;
	}
}