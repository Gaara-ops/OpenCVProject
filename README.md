# OpenProject
learn opencv

&nbsp;&nbsp;&nbsp;&nbsp;有了DirectX射线跟踪和现代gpu，射射线比以往任何时候都要**快**。然而，光线跟踪并不是**免费的**。至少在不久的将来，您可以假设每个像素最多有几束光线。这意味着混合光线光栅算法、抗混叠、去噪和重建都是快速实现高质量渲染的关键。本书中的其他工作提供了一些关于这些主题的想法，但是许多问题仍然没有得到解决。



配置着色器表指针和分派维度之后，使用新的命令列表函数 **SetPipelineState1()** 设置RTPSO，并使 **DispatchRays()** 生成光线。





参考文献

[1]   Benty, N. DirectX Raytracing Tutorials. https://github.com/NVIDIAGameWorks/DxrTutorials, 2018.   Accessed October 25, 2018.

[2] Benty, N., Yao, K.-H., Foley, T., Kaplanyan, A. S., Lavelle, C., Wyman, C., and Vijay, A. The Falcor Rendering Framework. https://github.com/NVIDIAGameWorks/Falcor, July 2017.

[3] Marrs, A. Introduction to DirectX Raytracing. https://github.com/acmarrs/IntroToDXR,2018. Accessed October 25, 2018.

[4] Marschner, S., and Shirley, P. Fundamentals of Computer Graphics, fourth ed. CRC Press, 2015.

[5] Microsoft. Programming Guide and Reference for HLSL. https://docs.microsoft.com/en-us/windows/desktop/direct3dhlsl/dx-graphics-hlsl. Accessed October 25,2018.

[6] Microsoft. D3D12 Raytracing Samples. https://github.com/Microsoft/DirectX-Graphics-Samples/tree/master/Samples/Desktop/D3D12Raytracing, 2018.Accessed October 25, 2018.

[7] Microsoft. DirectX Shader Compiler. https://github.com/Microsoft/DirectXShaderCompiler, 2018. Accessed October 30, 2018.

[8] NVIDIA. DirectX Raytracing Developer Blogs. https://devblogs.nvidia.com/tag/dxr/,2018. Accessed October 25, 2018.

[9] Shirley, P. Ray Tracing in One Weekend. Amazon Digital Services LLC, 2016. https://github.com/petershirley/raytracinginoneweekend.

[10] Suffern, K. Ray Tracing from the Ground Up. A K Peters, 2007.

[11] Wyman, C. A Gentle Introduction To DirectX Raytracing. http://cwyman.org/code/dxrTutors/dxr_tutors.md.html, 2018.

[12] Wyman, C., Hargreaves, S., Shirley, P., and Barré-Brisebois, C. Introduction to DirectX Raytracing.SIGGRAPH Courses, 2018. http://intro-to-dxr.cwyman.org, https://www.youtube.com/watch?v=Q1cuuepVNoY.