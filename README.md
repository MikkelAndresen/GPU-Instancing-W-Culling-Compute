# GPU-Instancing-W-Culling-Compute
This repo contains a basic system which generates matrices and renders them.
I've made this small project to familiarize myself with how jobs and burst can be used alongside URP, shaders, compute shaders and buffers.

The data gen script is called "TestDataGenerator.cs" and the renderer called "ManualIndirectInstanceTester.cs".
TestDataGenerator uses parallel burst jobs to generate the matrices. I've isolated the cost of this in its own profiler marker.

ManualIndirectInstanceTester copies the matrices from the generator to the GPU and is able to do so using multiple methods which can be selected during runtime from the inspector.
It utilizes a compute shader called MatrixFrustumCullingCompute.compute in order to do frustum culling.
It also calls the DrawMeshInstancedIndirect method using a material with the matrix buffer.
From the inspector of this script you can choose how the matrix data upload happens, when it happens and other render settings.
This settings all work during runtime, except for the Dimension and spacing (I might add that later) which is how many matrices are to be generated.

There are two shaders supported with this instancing system:
InstancedIndirectUnlit.shader and CustomLitShader.shader

The unlit is built from scratch for testing the most basic things, while the lit is derived from URP lit and is basically just hacked to take a matrix, color and index buffer.

There is also a few utility classes used for making it easier to use compute buffers inside the GenericComputeBuffer.cs file.