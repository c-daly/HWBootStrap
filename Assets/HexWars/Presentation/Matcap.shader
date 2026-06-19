Shader "HexWars/Matcap"
{
    Properties
    {
        _BaseColor ("Tint", Color) = (1,1,1,1)
        _Matcap ("Matcap", 2D) = "white" {}
        _Variation ("World Variation", Range(0,0.3)) = 0.1
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }

        Pass
        {
            Cull Off
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_Matcap);
            SAMPLER(sampler_Matcap);

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor;
                float4 _Matcap_ST;
                float _Variation;
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 matcapUV    : TEXCOORD0;
                float3 positionWS  : TEXCOORD1;
            };

            Varyings vert (Attributes IN)
            {
                Varyings OUT;
                OUT.positionWS = TransformObjectToWorld(IN.positionOS.xyz);
                OUT.positionHCS = TransformWorldToHClip(OUT.positionWS);
                float3 normalWS = TransformObjectToWorldNormal(IN.normalOS);
                float3 normalVS = mul((float3x3)UNITY_MATRIX_V, normalWS);
                OUT.matcapUV = normalVS.xy * 0.5 + 0.5;
                return OUT;
            }

            half4 frag (Varyings IN) : SV_Target
            {
                // flat per-face shading, no grain noise (it read as fuzz)
                half3 mc = SAMPLE_TEXTURE2D(_Matcap, sampler_Matcap, IN.matcapUV).rgb;
                half lum = mc.r;
                half3 tinted = mc * _BaseColor.rgb;             // colored body
                half spec = saturate((lum - 0.85) / 0.15) * 0.35; // subtle white sheen only at the brightest
                half3 col = lerp(tinted, half3(lum, lum, lum), spec);
                return half4(col, 1.0);
            }
            ENDHLSL
        }
    }
}
