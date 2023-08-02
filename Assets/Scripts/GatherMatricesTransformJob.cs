using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Jobs;
using Plane = MathUtil.Plane;

public struct GatherMatricesTransformJob : IJobParallelForTransform
{
	[WriteOnly]
	private NativeArray<float3x4> matrices;

	public GatherMatricesTransformJob(NativeArray<float3x4> matrices) => this.matrices = matrices;

	public void Execute(int index, [ReadOnly]TransformAccess transform)
	{
		float3x3 rotMatrix = new float3x3(transform.rotation);
		matrices[index] = new float3x4(rotMatrix.c0, rotMatrix.c1, rotMatrix.c2, transform.position);
	}
}