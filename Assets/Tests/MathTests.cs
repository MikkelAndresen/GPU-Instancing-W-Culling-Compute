using NUnit.Framework;
using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using static MathUtil;
using Plane = MathUtil.Plane;

public class MathTests
{
	[Test]
	public void AABB4ClassifyAgainstPlane()
	{
		AABB inPlane = new AABB(0, 1);
		AABB frontOfPlane = new AABB(10, 1);
		AABB behindPlane = new AABB(-10, 1);
		AABB4 bounds = new AABB4(inPlane, frontOfPlane, inPlane, behindPlane);
		Plane p = new MathUtil.Plane(math.forward(), 1);
		float4 result = bounds.ClassifyAgainstPlane(p);

		AssertResult();

		// We test against the other classify method which is not vectorized
		for (int i = 0; i < 4; i++)
			result[i] = bounds[i].ClassifyAgainstPlane(p);

		AssertResult();

		void AssertResult()
		{
			Assert.AreEqual(result.x, 0);
			Assert.AreEqual(result.y, 10);
			Assert.AreEqual(result.z, 0);
			Assert.AreEqual(result.w, -8);
		}
	}

	[Test]
	public void Dot4Test()
	{
		float3x4 vectors = new float3x4(math.right(), math.forward(), math.up(), -math.forward());
		float4x3 flipped = math.transpose(vectors);
		float4 dot4 = MathUtil.Dot4(flipped.c0, flipped.c1, flipped.c2, vectors[1]);
		float4 dotBasic = default;
		for (int i = 0; i < 4; i++)
			dotBasic[i] = math.dot(vectors[i], vectors[1]);

		Assert.AreEqual(dotBasic, dot4);
	}

	[Test]
	public void AABB4Frustum()
	{
		Camera cam = Camera.main;
		Transform camTrans = cam.transform;
		NativeArray<Plane> frustum = new NativeArray<Plane>(6, Allocator.Persistent);
		UnityEngine.Plane[] f = GeometryUtility.CalculateFrustumPlanes(cam);
		for (int i = 0; i < f.Length; i++)
			frustum[i] = new Plane(f[i]);
		
		float halfFov = cam.fieldOfView / 2;
		Vector3 frustumVector = camTrans.forward;
		frustumVector = Quaternion.Euler(0, halfFov, 0) * frustumVector;

		AABB inFrustum = new AABB(camTrans.position + new Vector3(0,0,1), 1);
		AABB onFrustum = new AABB(frustumVector * 2f, 1);
		AABB behindFrustum = new AABB(new float3(0,0,-5), 1);
		AABB slightlyOutOfFrustum = new AABB(onFrustum.Center + new float3(0,0,-2f), 1);
		AABB4 bounds = new AABB4(inFrustum, onFrustum, behindFrustum, slightlyOutOfFrustum);

		bool4 result = bounds.IsBoundsInFrustum(frustum);

		frustum.Dispose();

		Debug.Log(result);
		Assert.IsTrue(result[0]);
		Assert.IsTrue(result[1]);
		Assert.IsFalse(result[2]);
		Assert.IsTrue(result[3]);
	}
}