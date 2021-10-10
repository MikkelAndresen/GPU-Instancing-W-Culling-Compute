Shader "Unlit/InstancedIndirectUnlit"
{
	Properties
	{
		
	}
	SubShader
	{
		Tags { "RenderType"="Opaque" }
		LOD 100

		Pass
		{
			CGPROGRAM
			// Upgrade NOTE: excluded shader from OpenGL ES 2.0 because it uses non-square matrices
			#pragma exclude_renderers gles
			#pragma vertex vert
			#pragma fragment frag
			// make fog work
			#pragma multi_compile_fog
			#pragma target 4.5
			#include "UnityCG.cginc"
			#include "UnityLightingCommon.cginc"
			#include "AutoLight.cginc"

			struct CustomMatrix
			{
				float4x4 mat;
			};

#if SHADER_TARGET >= 45
		   StructuredBuffer<uint> indexBuffer;
		   StructuredBuffer<float3x4> matrixBuffer;
		   StructuredBuffer<float4> colorBuffer;
#endif
			
			struct v2f
			{
				float4 vertex : SV_POSITION;
				float2 uv : TEXCOORD0;
				uint index : TEXCOORD1;
				SHADOW_COORDS(4)
			};

			v2f vert (appdata_full v, uint instanceID : SV_InstanceID)
			{
				uint index = indexBuffer[instanceID];
				float3x4 mat = matrixBuffer[index];
				//float3x4 mat = matrixBuffer[1];

				// Testing for directly fetching matrices
				//float3x4 mat = matrixBuffer[instanceID];

				//float4x4 mat = data.mat;
				//float4x4 mat4 = 
				//{
				//	float4(1,0,0,instanceID),
				//	float4(0,1,0,instanceID),
				//	float4(0,0,1,instanceID),
				//	float4(1,1,1,1)
				//};

				// Convert to float4x4 and add a scale of 1
				float4x4 mat4 = 
				{
					mat[0],
					mat[1],
					mat[2],
					float4(1,1,1,1)
				};
				
				v.vertex = mul(mat4, v.vertex);
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				//o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
				o.uv = v.texcoord;
				o.index = index;
				UNITY_TRANSFER_FOG(o, o.vertex);
				return o;
			}

			fixed4 frag (v2f i) : SV_Target
			{
				//fixed4 col = i.data.color;
				// apply fog
				//UNITY_APPLY_FOG(i.fogCoord, col);
				return colorBuffer[i.index];
			}
			ENDCG
		}
	}
}