using System.Runtime.CompilerServices;
using Unity.Mathematics;

public static class MathUtil
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int3 Get3DIndex(int index, int dimX, int dimY) => new int3(index % dimX, (index / dimX) % dimY, index / (dimY * dimX));
}