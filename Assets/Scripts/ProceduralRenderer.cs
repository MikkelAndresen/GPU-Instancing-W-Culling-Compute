using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(FrustrumFilterTransformJobSystem))]
public class ProceduralRenderer : MonoBehaviour
{
	[SerializeField] private Mesh mesh;
	[SerializeField] private ShadowCastingMode shadowCastingMode = ShadowCastingMode.On;
	[SerializeField] private bool receiveShadows = true;
	[SerializeField] private Material mat;
	[SerializeField] private LightProbeUsage lightProbeUsage = LightProbeUsage.BlendProbes;

	private FrustrumFilterTransformJobSystem frustumCuller;
	private readonly Bounds bounds = new Bounds(Vector3.zero, Vector3.one * 10000);
	protected static readonly int materialMatrixBufferID = Shader.PropertyToID("matrixBuffer");

	private void Awake()
	{
		frustumCuller = GetComponent<FrustrumFilterTransformJobSystem>();
		frustumCuller.AutoCompleteInLateUpdate = false;
		mat.SetBuffer(materialMatrixBufferID, frustumCuller.MatrixBuffer);
	}

	private void OnEnable() => RenderPipelineManager.beginCameraRendering += RenderCamera;

	private void OnDisable() => RenderPipelineManager.beginCameraRendering -= RenderCamera;

	private readonly ProfilerMarker renderMarker = new("Render");

	private void RenderCamera(ScriptableRenderContext arg1, Camera cam)
	{
		// Will complete job if running
		frustumCuller.CompleteFilterJob();
		if (frustumCuller.FilteredCount == 0)
			return;

		renderMarker.Begin();
		Graphics.DrawMeshInstancedProcedural(
			mesh,
			submeshIndex: 0,
			mat,
			bounds,
			count: frustumCuller.FilteredCount,
			properties: null,
			shadowCastingMode,
			receiveShadows,
			gameObject.layer,
			camera: cam,
			lightProbeUsage,
			lightProbeProxyVolume: null
		);
		renderMarker.End();
	}
}