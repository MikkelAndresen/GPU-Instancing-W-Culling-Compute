using System;
using UnityEngine;
using Unity.Mathematics;
using System.Collections.Generic;
using UnityEngine.Profiling;

namespace StagPoint.Math
{
	public struct Bounds
	{
		public Bounds(float3 center, float3 extents)
		{
			this.center = center;
			this.extents = extents;
			Max = center + extents;
			Min = center - extents;
		}

		private float3 center;
		public float3 Center 
		{
			get => center;
			set 
			{
				center = value;
				InvalidateMinMax();
			}
		}

		private float3 extents;
		public float3 Extents 
		{
			get => extents;
			set
			{
				extents = value;
				InvalidateMinMax();
			}
		}

		public float3 Min { get; private set; }
		public float3 Max { get; private set; }

		private void InvalidateMinMax()
		{
			Max = Center + Extents;
			Min = Center - Extents;
		}

		public bool Contains(float3 point) =>
			point.x >= Min.x && point.y >= Min.y && point.z >= Min.z &&
			point.x <= Max.x && point.y <= Max.y && point.z <= Max.z;
	}

	[System.Serializable]
	public class BoundingFrustum
	{
		public enum IntersectionType
		{
			Disjoint,
			Intersects,
			Contains
		}

		public static readonly float3 forward = new float3(0, 0, 1);
		public static readonly float3 right = new float3(1, 0, 0);
		public static readonly float3 up = new float3(0, 1, 0);

		#region Public fields 

		/// <summary>
		/// The number of planes in the frustum.
		/// </summary>
		public const int PlaneCount = 6;

		/// <summary>
		/// The number of corner points in the frustum.
		/// </summary>
		public const int CornerCount = 8;

		/// <summary>
		/// Returns the current position of the frustum
		/// </summary>
		public float3 Position { get; private set; }

		#endregion

		#region Public Properties

		public Plane[] Planes => _planes;

		#endregion

		#region Private variables

		/// <summary>
		/// Ordering: [0] = Far Bottom Left, [1] = Far Top Left, [2] = Far Top Right, [3] = Far Bottom Right, 
		/// [4] = Near Bottom Left, [5] = Near Top Left, [6] = Near Top Right, [7] = Near Bottom Right
		/// </summary>
		[SerializeField]
		private float3[] _corners = new float3[CornerCount];

		/// <summary>
		/// Defines the set of planes that bound the camera's frustum. All plane normals point to the inside of the 
		/// frustum.
		/// Ordering: [0] = Left, [1] = Right, [2] = Down, [3] = Up, [4] = Near, [5] = Far
		/// </summary>
		[SerializeField]
		private Plane[] _planes = new Plane[PlaneCount];

		/// <summary>
		/// Caches the absolute values of plane normals for re-use during frustum culling of multiple AABB instances
		/// </summary>
		[SerializeField]
		private float3[] _absNormals = new float3[PlaneCount];

		/// <summary>
		/// Caching the plane normals allows the culling code to avoid calling property getters on the Plane instances
		/// </summary>
		[SerializeField]
		private float3[] _planeNormal = new float3[PlaneCount];

		/// <summary>
		/// Caching the plane distances allows the culling code to avoid calling property getters on the Plane instances
		/// </summary>
		[SerializeField]
		private float[] _planeDistance = new float[PlaneCount];

		#endregion

		#region Public functions

		/// <summary>
		/// Extracts the frustum corners. The destination array must contain space for no less than CornerCount elements.
		/// Ordering: [0] = Far Bottom Left, [1] = Far Top Left, [2] = Far Top Right, [3] = Far Bottom Right, [4] = Camera Position
		/// </summary>
		public void GetCorners(float3[] outCorners)
		{
			if (outCorners == null || outCorners.Length < CornerCount)
			{
				throw new InvalidOperationException("Destination array is null or too small");
			}

			Array.Copy(_corners, outCorners, CornerCount);
		}

		/// <summary>
		/// Extracts the frustum planes. The destination array must contain space for no less than PlaneCount elements.
		/// Ordering: [0] = Left, [1] = Right, [2] = Down, [3] = Up, [4] = Near, [5] = Far
		/// </summary>
		public void GetPlanes(Plane[] outPlanes)
		{
			if (outPlanes == null || outPlanes.Length < PlaneCount)
			{
				throw new InvalidOperationException("Destination array is null or too small");
			}

			Array.Copy(_planes, outPlanes, PlaneCount);
		}

		/// <summary>
		/// Update the bounding frustum from the current camera settings
		/// </summary>
		public void Update(Camera camera)
		{
			Update(camera, camera.farClipPlane);
		}

		/// <summary>
		/// Update the bounding frustum from the current camera settings
		/// </summary>
		public void Update(Camera camera, float farClipPlane)
		{
			Profiler.BeginSample("Update Frustrum");

			var camTransform = camera.transform;
			float3 position = new float3(
				camTransform.position.x,
				camTransform.position.y,
				camTransform.position.z
			);
			var orientation = new quaternion(
				camTransform.rotation.x,
				camTransform.rotation.y,
				camTransform.rotation.z,
				camTransform.rotation.w);

			this.Position = position;
			var forward = math.mul(orientation, BoundingFrustum.forward);

			if (camera.orthographic)
			{
				calculateFrustumCornersOrthographic(camera);
			}
			else
			{
				calculateFrustumCornersPerspective(
					ref position,
					ref orientation,
					camera.fieldOfView,
					camera.nearClipPlane,
					camera.farClipPlane,
					camera.aspect
				);
			}

			// CORNERS:
			// [0] = Far Bottom Left,  [1] = Far Top Left,  [2] = Far Top Right,  [3] = Far Bottom Right, 
			// [4] = Near Bottom Left, [5] = Near Top Left, [6] = Near Top Right, [7] = Near Bottom Right

			// PLANES:
			// Ordering: [0] = Left, [1] = Right, [2] = Down, [3] = Up, [4] = Near, [5] = Far

			_planes[0] = new Plane(_corners[4], _corners[1], _corners[0]);
			_planes[1] = new Plane(_corners[6], _corners[3], _corners[2]);
			_planes[2] = new Plane(_corners[7], _corners[0], _corners[3]);
			_planes[3] = new Plane(_corners[5], _corners[2], _corners[1]);
			_planes[4] = new Plane(forward, position + forward * camera.nearClipPlane);
			_planes[5] = new Plane(-forward, position + forward * farClipPlane);

			for (int i = 0; i < PlaneCount; i++)
			{
				var plane = _planes[i];
				var normal = plane.normal;

				_absNormals[i] = new float3(math.abs(normal.x), math.abs(normal.y), math.abs(normal.z));
				_planeNormal[i] = normal;
				_planeDistance[i] = plane.distance;
			}

			Profiler.EndSample();
		}

		/// <summary>
		/// Update the bounding frustum
		/// </summary>
		public void Update(float3 position, quaternion orientation, float fov, float nearClipPlane, float farClipPlane, float aspect)
		{
			this.Position = position;

			calculateFrustumCornersPerspective(ref position, ref orientation, fov, nearClipPlane, farClipPlane, aspect);

			var forward = math.mul(orientation, BoundingFrustum.forward);

			// CORNERS:
			// [0] = Far Bottom Left,  [1] = Far Top Left,  [2] = Far Top Right,  [3] = Far Bottom Right, 
			// [4] = Near Bottom Left, [5] = Near Top Left, [6] = Near Top Right, [7] = Near Bottom Right

			// PLANES:
			// Ordering: [0] = Left, [1] = Right, [2] = Down, [3] = Up, [4] = Near, [5] = Far

			_planes[0] = new Plane(_corners[4], _corners[1], _corners[0]);
			_planes[1] = new Plane(_corners[6], _corners[3], _corners[2]);
			_planes[2] = new Plane(_corners[7], _corners[0], _corners[3]);
			_planes[3] = new Plane(_corners[5], _corners[2], _corners[1]);
			_planes[4] = new Plane(forward, position + forward * nearClipPlane);
			_planes[5] = new Plane(-forward, position + forward * farClipPlane);

			for (int i = 0; i < PlaneCount; i++)
			{
				var plane = _planes[i];
				var normal = plane.normal;

				_absNormals[i] = new float3(math.abs(normal.x), math.abs(normal.y), math.abs(normal.z));
				_planeNormal[i] = normal;
				_planeDistance[i] = plane.distance;
			}
		}

		//public bool Contains(ref Bounds bounds)
		//{
		//	for (int i = 0; i < PlaneCount; i++)
		//	{
		//		var normal = _planeNormal[i];
		//		var distance = _planeDistance[i];

		//		float dist = normal.x * point.x + normal.y * point.y + normal.z * point.z + distance;

		//		if (dist < 0f)
		//		{
		//			return false;
		//		}
		//	}

		//	return true;
		//}

		/// <summary>
		/// Returns true if the frustum contains the specified point
		/// </summary>
		public bool Contains(ref float3 point)
		{
			Profiler.BeginSample("Contains");
			for (int i = 0; i < PlaneCount; i++)
			{
				var normal = _planeNormal[i];
				var distance = _planeDistance[i];

				float dist = normal.x * point.x + normal.y * point.y + normal.z * point.z + distance;

				if (dist < 0f)
				{
					Profiler.EndSample();
					return false;
				}
			}
			Profiler.EndSample();
			return true;
		}

		/// <summary>
		/// Returns the type of intersection (if any) between the frustum and the sphere
		/// </summary>
		/// <param name="center">The world position of the sphere</param>
		/// <param name="radius">The radius of the sphere</param>
		public IntersectionType GetSphereIntersection(ref float3 center, float radius, float frustumPadding = 0)
		{
			var intersecting = false;

			for (int i = 0; i < PlaneCount; i++)
			{
				var normal = _planeNormal[i];
				var distance = _planeDistance[i];

				float dist = normal.x * center.x + normal.y * center.y + normal.z * center.z + distance;

				if (dist < -radius - frustumPadding)
				{
					return IntersectionType.Disjoint;
				}

				intersecting |= (dist <= radius);
			}

			return intersecting ? IntersectionType.Intersects : IntersectionType.Contains;
		}

		/// <summary>
		/// Returns the type of intersection (if any) between the frustum and the sphere
		/// </summary>
		public IntersectionType GetSphereIntersection(ref BoundingSphere sphere, float frustumPadding = 0)
		{
			var intersecting = false;

			var center = sphere.position;
			var radius = sphere.radius;

			for (int i = 0; i < PlaneCount; i++)
			{
				var normal = _planeNormal[i];
				var distance = _planeDistance[i];

				float dist = normal.x * center.x + normal.y * center.y + normal.z * center.z + distance;

				if (dist < -radius - frustumPadding)
				{
					return IntersectionType.Disjoint;
				}

				intersecting |= (dist <= radius);
			}

			return intersecting ? IntersectionType.Intersects : IntersectionType.Contains;
		}

		/// <summary>
		/// Iterates through each sphere in the array and sets the Result field to the result of the sphere/frustum intersection test
		/// This function is intended primarily for use with static geometry (or quadtrees, etc) where the bounding volumes will not 
		/// be updated frequently, but the frustum will. 
		/// </summary>
		public void CullSpheres(CullingSphere[] spheres, int sphereCount)
		{
			float3 planeNormal = float3.zero;

			var planeNormal0 = _planeNormal[0];
			var planeNormal1 = _planeNormal[1];
			var planeNormal2 = _planeNormal[2];
			var planeNormal3 = _planeNormal[3];
			var planeNormal4 = _planeNormal[4];
			var planeNormal5 = _planeNormal[5];

			var planeDistance0 = _planeDistance[0];
			var planeDistance1 = _planeDistance[1];
			var planeDistance2 = _planeDistance[2];
			var planeDistance3 = _planeDistance[3];
			var planeDistance4 = _planeDistance[4];
			var planeDistance5 = _planeDistance[5];

			for (int si = 0; si < sphereCount; si++)
			{
				var sphere = spheres[si];
				var center = sphere.SphereCenter;
				var radius = sphere.SphereRadius;

				bool outOfFrustum = false;

				outOfFrustum = outOfFrustum || (planeNormal0.x * center.x + planeNormal0.y * center.y + planeNormal0.z * center.z + planeDistance0) < -radius;
				outOfFrustum = outOfFrustum || (planeNormal1.x * center.x + planeNormal1.y * center.y + planeNormal1.z * center.z + planeDistance1) < -radius;
				outOfFrustum = outOfFrustum || (planeNormal2.x * center.x + planeNormal2.y * center.y + planeNormal2.z * center.z + planeDistance2) < -radius;
				outOfFrustum = outOfFrustum || (planeNormal3.x * center.x + planeNormal3.y * center.y + planeNormal3.z * center.z + planeDistance3) < -radius;
				outOfFrustum = outOfFrustum || (planeNormal4.x * center.x + planeNormal4.y * center.y + planeNormal4.z * center.z + planeDistance4) < -radius;
				outOfFrustum = outOfFrustum || (planeNormal5.x * center.x + planeNormal5.y * center.y + planeNormal5.z * center.z + planeDistance5) < -radius;

				spheres[si].IsInFrustum = !outOfFrustum;
			}
		}

		/// <summary>
		/// Returns the type of intersection (if any) between the frustum and the sphere
		/// </summary>
		/// <param name="center">The world position of the sphere</param>
		/// <param name="radius">The radius of the sphere</param>
		public bool IntersectsSphere(ref float3 center, float radius, float frustumPadding = 0)
		{
			for (int i = 0; i < PlaneCount; i++)
			{
				var normal = _planeNormal[i];
				var distance = _planeDistance[i];

				float dist = normal.x * center.x + normal.y * center.y + normal.z * center.z + distance;

				if (dist < -radius - frustumPadding)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// Returns the type of intersection (if any) between the frustum and the sphere
		/// </summary>
		/// <param name="sphere">The sphere to check</param>
		public bool IntersectsSphere(ref BoundingSphere sphere, float frustumPadding = 0)
		{
			var center = sphere.position;

			for (int i = 0; i < PlaneCount; i++)
			{
				var normal = _planeNormal[i];
				var distance = _planeDistance[i];

				float dist = normal.x * center.x + normal.y * center.y + normal.z * center.z + distance;

				if (dist < -sphere.radius - frustumPadding)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// Returns TRUE if the box and frustum intersect
		/// </summary>
		public bool IntersectsBox(ref Bounds box, float frustumPadding = 0)
		{
			// Exit early if the box contains the frustum origin
			if (box.Contains(_corners[CornerCount - 1]))
			{
				return true;
			}

			var center = box.Center;
			var extents = box.Extents;

			for (int i = 0; i < PlaneCount; i++)
			{
				var abs = _absNormals[i];

				var planeNormal = _planeNormal[i];
				var planeDistance = _planeDistance[i];

				float r = extents.x * abs.x + extents.y * abs.y + extents.z * abs.z;
				float s = planeNormal.x * center.x + planeNormal.y * center.y + planeNormal.z * center.z;

				if (s + r < -planeDistance - frustumPadding)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// Returns the type of intersection (if any) between the bounding box and the frustum
		/// </summary>
		public IntersectionType GetBoxIntersection(ref Bounds box, float frustumPadding = 0)
		{
			var center = box.Center;
			var extents = box.Extents;

			var intersecting = false;

			for (int i = 0; i < PlaneCount; i++)
			{
				var abs = _absNormals[i];

				var planeNormal = _planeNormal[i];
				var planeDistance = _planeDistance[i];

				float r = extents.x * abs.x + extents.y * abs.y + extents.z * abs.z;
				float s = planeNormal.x * center.x + planeNormal.y * center.y + planeNormal.z * center.z;

				if (s + r < -planeDistance - frustumPadding)
				{
					return IntersectionType.Disjoint;
				}

				intersecting |= (s - r <= -planeDistance);
			}

			return intersecting ? IntersectionType.Intersects : IntersectionType.Contains;
		}

		/// <summary>
		/// Iterates through each box in the boxes array and sets the Result field to the result of the box/frustum intersection test.
		/// This function is intended primarily for use with static geometry (or quadtrees, etc) where the bounding volumes will not 
		/// be updated frequently, but the frustum will. 
		/// </summary>
		public void CullBoxes(CullingBox[] boxes, int boxCount)
		{
			var abs0 = _absNormals[0];
			var abs1 = _absNormals[1];
			var abs2 = _absNormals[2];
			var abs3 = _absNormals[3];
			var abs4 = _absNormals[4];
			var abs5 = _absNormals[5];

			var planeNormal0 = _planeNormal[0];
			var planeNormal1 = _planeNormal[1];
			var planeNormal2 = _planeNormal[2];
			var planeNormal3 = _planeNormal[3];
			var planeNormal4 = _planeNormal[4];
			var planeNormal5 = _planeNormal[5];

			var planeDistance0 = _planeDistance[0];
			var planeDistance1 = _planeDistance[1];
			var planeDistance2 = _planeDistance[2];
			var planeDistance3 = _planeDistance[3];
			var planeDistance4 = _planeDistance[4];
			var planeDistance5 = _planeDistance[5];

			for (int bi = 0; bi < boxCount; bi++)
			{
				var box = boxes[bi];
				var center = box.BoxCenter;
				var extents = box.BoxExtents;

				bool outOfFrustum = false;

				outOfFrustum = outOfFrustum || (
					(extents.x * abs0.x + extents.y * abs0.y + extents.z * abs0.z) +
					(planeNormal0.x * center.x + planeNormal0.y * center.y + planeNormal0.z * center.z)) < -planeDistance0;

				outOfFrustum = outOfFrustum || (
					(extents.x * abs1.x + extents.y * abs1.y + extents.z * abs1.z) +
					(planeNormal1.x * center.x + planeNormal1.y * center.y + planeNormal1.z * center.z)) < -planeDistance1;

				outOfFrustum = outOfFrustum || (
					(extents.x * abs2.x + extents.y * abs2.y + extents.z * abs2.z) +
					(planeNormal2.x * center.x + planeNormal2.y * center.y + planeNormal2.z * center.z)) < -planeDistance2;

				outOfFrustum = outOfFrustum || (
					(extents.x * abs3.x + extents.y * abs3.y + extents.z * abs3.z) +
					(planeNormal3.x * center.x + planeNormal3.y * center.y + planeNormal3.z * center.z)) < -planeDistance3;

				outOfFrustum = outOfFrustum || (
					(extents.x * abs4.x + extents.y * abs4.y + extents.z * abs4.z) +
					(planeNormal4.x * center.x + planeNormal4.y * center.y + planeNormal4.z * center.z)) < -planeDistance4;

				outOfFrustum = outOfFrustum || (
					(extents.x * abs5.x + extents.y * abs5.y + extents.z * abs5.z) +
					(planeNormal5.x * center.x + planeNormal5.y * center.y + planeNormal5.z * center.z)) < -planeDistance5;

				boxes[bi].IsInFrustum = !outOfFrustum;
			}
		}

		/// <summary>
		/// Returns TRUE if the oriented bounding box and frustum intersect
		/// </summary>
		/// <param name="box">The bounding box to test. Note: box.center is expected to be in world coordinates</param>
		/// <param name="right">The horizontal local coordinate axis (equivalent to Transform.right)</param>
		/// <param name="up">The vertical local coordinate axis (equivalent to Transform.up)</param>
		/// <param name="forward">The forward local coordinate axis (equivalent to Transform.forward)</param>
		/// <returns></returns>
		public bool IntersectsOrientedBox(ref Bounds box, ref float3 right, ref float3 up, ref float3 forward, float frustumPadding = 0)
		{
			var center = box.Center;
			var extents = box.Extents;

			for (int i = 0; i < PlaneCount; i++)
			{
				var planeNormal = _planeNormal[i];
				var planeDistance = _planeDistance[i];

				float r =
					extents.x * math.abs(math.dot(planeNormal, right)) +
					extents.y * math.abs(math.dot(planeNormal, up)) +
					extents.z * math.abs(math.dot(planeNormal, forward));

				float s = planeNormal.x * center.x + planeNormal.y * center.y + planeNormal.z * center.z;

				if (s + r < -planeDistance - frustumPadding)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// Returns the type of intersection (if any) of an oriented bounding box and the frustum.
		/// </summary>
		/// <param name="box">The bounding box to test. Note: box.center is expected to be in world coordinates</param>
		/// <param name="right">The horizontal local coordinate axis (equivalent to Transform.right)</param>
		/// <param name="up">The vertical local coordinate axis (equivalent to Transform.up)</param>
		/// <param name="forward">The forward local coordinate axis (equivalent to Transform.forward)</param>
		/// <returns></returns>
		public IntersectionType GetOrientedBoxIntersection(ref Bounds box, ref float3 right, ref float3 up, ref float3 forward, float frustumPadding = 0)
		{
			var center = box.Center;
			var extents = box.Extents;

			var intersecting = false;

			for (int i = 0; i < PlaneCount; i++)
			{
				var planeNormal = _planeNormal[i];
				var planeDistance = _planeDistance[i];

				float r =
					extents.x * math.abs(math.dot(planeNormal, right)) +
					extents.y * math.abs(math.dot(planeNormal, up)) +
					extents.z * math.abs(math.dot(planeNormal, forward));

				float s = planeNormal.x * center.x + planeNormal.y * center.y + planeNormal.z * center.z;

				if (s + r < -planeDistance - frustumPadding)
				{
					return IntersectionType.Disjoint;
				}

				intersecting |= (s - r <= -planeDistance);
			}

			return intersecting ? IntersectionType.Intersects : IntersectionType.Contains;
		}

		#endregion

		#region Private functions 

		private void calculateFrustumCornersOrthographic(Camera camera)
		{
			Profiler.BeginSample(nameof(calculateFrustumCornersOrthographic));

			var camTransform = camera.transform;
			var position = camTransform.position;
			var orientation = camTransform.rotation;
			var farClipPlane = camera.farClipPlane;
			var nearClipPlane = camera.nearClipPlane;

			var forward = orientation * BoundingFrustum.forward;
			var right = orientation * BoundingFrustum.right * camera.orthographicSize * camera.aspect;
			var up = orientation * BoundingFrustum.up * camera.orthographicSize;

			// CORNERS:
			// [0] = Far Bottom Left,  [1] = Far Top Left,  [2] = Far Top Right,  [3] = Far Bottom Right, 
			// [4] = Near Bottom Left, [5] = Near Top Left, [6] = Near Top Right, [7] = Near Bottom Right

			_corners[0] = position + forward * farClipPlane - up - right;
			_corners[1] = position + forward * farClipPlane + up - right;
			_corners[2] = position + forward * farClipPlane + up + right;
			_corners[3] = position + forward * farClipPlane - up + right;
			_corners[4] = position + forward * nearClipPlane - up - right;
			_corners[5] = position + forward * nearClipPlane + up - right;
			_corners[6] = position + forward * nearClipPlane + up + right;
			_corners[7] = position + forward * nearClipPlane - up + right;

			Profiler.EndSample();
		}

		private void calculateFrustumCornersPerspective(ref float3 position, ref quaternion orientation, float fov, float nearClipPlane, float farClipPlane, float aspect)
		{
			Profiler.BeginSample(nameof(calculateFrustumCornersPerspective));

			float fovWHalf = fov * 0.5f;

			float3 toRight = right * Mathf.Tan(fovWHalf * Mathf.Deg2Rad) * aspect;
			float3 toTop = up * Mathf.Tan(fovWHalf * Mathf.Deg2Rad);

			float3 topLeft = (forward - toRight + toTop);
			float camScale = math.length(topLeft) * farClipPlane;

			math.normalize(topLeft);
			topLeft *= camScale;

			float3 topRight = (forward + toRight + toTop);
			math.normalize(topRight);
			topRight *= camScale;

			float3 bottomRight = (forward + toRight - toTop);
			math.normalize(bottomRight);
			bottomRight *= camScale;

			float3 bottomLeft = (forward - toRight - toTop);
			math.normalize(bottomLeft);
			bottomLeft *= camScale;

			_corners[0] = position + math.mul(orientation, bottomLeft);
			_corners[1] = position + math.mul(orientation, topLeft);
			_corners[2] = position + math.mul(orientation, topRight);
			_corners[3] = position + math.mul(orientation, bottomRight);

			topLeft = (forward - toRight + toTop);
			camScale = math.length(topLeft) * nearClipPlane;

			math.normalize(topLeft);
			topLeft *= camScale;

			topRight = (forward + toRight + toTop);
			math.normalize(topRight);
			topRight *= camScale;

			bottomRight = (forward + toRight - toTop);
			math.normalize(bottomRight);
			bottomRight *= camScale;

			bottomLeft = (forward - toRight - toTop);
			math.normalize(bottomLeft);
			bottomLeft *= camScale;

			_corners[4] = position + math.mul(orientation, bottomLeft);
			_corners[5] = position + math.mul(orientation, topLeft);
			_corners[6] = position + math.mul(orientation, topRight);
			_corners[7] = position + math.mul(orientation, bottomRight);

			Profiler.EndSample();
		}

		#endregion

		#region Nested types

		// When culling large numbers of static volumes per frame, it can be faster and more efficient to store just their 
		// bounding volume representations in a single indexed array, together with the culling results. This allows for 
		// extremely fast brute-force culling of large numbers of objects without the need to recursively traverse hierarchical 
		// spatial partition structures. This can in some particular cases actually be significantly faster.
		// This was implemented for a specific use case in my own code and YMMV, so profile rigorously and make no assumptions.

		public struct CullingBox
		{
			public float3 BoxCenter;
			public float3 BoxExtents;
			public bool IsInFrustum;
		}

		public struct CullingSphere
		{
			public float3 SphereCenter;
			public float SphereRadius;
			public bool IsInFrustum;
		}

		#endregion
	}
}