#include <stdio.h>
#include <vector>
#include <ctime>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <string>
#include <cuda_runtime.h>
#include <helper_functions.h>
using namespace std;

#define PI 3.1415926

//C++实现的split函数，注意：当字符串为空时，也会返回一个空字符串
vector<string> split(string s, string delim)
{
    vector<string> ret;
    size_t last = 0;
    size_t index=s.find_first_of(delim,last);
    while (index!=string::npos)
    {
        ret.push_back(s.substr(last,index-last));
    	last=index+1;
    	index=s.find_first_of(delim,last);
    }
    if (index-last>0)
    {
    	ret.push_back(s.substr(last,index-last));
    }
    return ret;
};

//方向向量类，包含三个方向的值
class dirVector
{
    private:
        float X,Y,Z;
    public:
        __device__ __host__ dirVector(float x=0, float y=0, float z=0)
        {
            X = x;
            Y = y;
            Z = z;
        }

        float getX() { return X;}
        float getY() { return Y;}
        float getZ() { return Z;}

        void setX(float x) { X = x;}
        void setY(float y) { Y = y;}
        void setZ(float z) { Z = z;}
        //计算两个向量叉乘
        __device__ __host__ dirVector chaCheng(dirVector tmp)
        {
            return dirVector((Y*tmp.Z - Z*tmp.Y), (Z*tmp.X - X*tmp.Z), (X*tmp.Y - Y*tmp.X));
        }
        //计算两个向量点乘
        __device__ __host__ float dianCheng(dirVector tmp)
        {
            return X*tmp.X + Y*tmp.Y + Z*tmp.Z;;
        }
};

//三维空间点的类，包含三个浮点数，分别是x,y,z坐标
class point
{
    private:
        float X,Y,Z;
    public:
        __device__ __host__ point(float x=0, float y=0, float z=0)
        {
            X = x;
            Y = y;
            Z = z;
        }
        
        float getX() { return X;}
        float getY() { return Y;}
        float getZ() { return Z;}

        void setX(float x) { X = x;}
        void setY(float y) { Y = y;}
        void setZ(float z) { Z = z;}
   
        //判断一个点是否在一个三角面中
        bool isOnTheFace(point a, point b, point c)
        {
            if ( (X == a.getX() && Y == a.getY() && Z == a.getZ()) || (X == b.getX() && Y == b.getY() && Z == b.getZ()) || (X == c.getX() && Y == c.getY() && Z == c.getZ()))
            {
                return true;
            }
            else
            {
                dirVector ab,ad,bc,bd,cd,ca;
                ab = dirVector(b.getX() - a.getX(), b.getY() - a.getY(), b.getZ() - a.getZ());
                ad = dirVector(X - a.getX(), Y - a.getY(), Z - a.getZ());
                bc = dirVector(c.getX() - b.getX(), c.getY() - b.getY(), c.getZ() - b.getZ());
                bd = dirVector(X - b.getX(), Y - b.getY(), Z - b.getZ());
                cd = dirVector(X - c.getX(), Y - c.getY(), Z - c.getZ());
                ca = dirVector(a.getX() - c.getX(), a.getY() - c.getY(), a.getZ() - c.getZ());

                dirVector adXab,bdXbc,cdXca;
                adXab = ad.chaCheng(ab);
                bdXbc = bd.chaCheng(bc);
                cdXca = cd.chaCheng(ca);

                float m,n,l;
                m = adXab.dianCheng(bdXbc);
                n = bdXbc.dianCheng(cdXca);
                l = cdXca.dianCheng(adXab);

                if (m >= 0 && n >= 0 && l >= 0) { return true; }
                else { return false; }
            }
        }
};

//三角面类，包含三个point类型的成员，表示三位空间的一个三角面
class face
{
    private:
        point A,B,C;
    public:
        face(point a=point(0,0,0), point b=point(0,0,0), point c=point(0,0,0))
        {
            A = a;
            B = b;
            C = c;
        }

        point getA() { return A;}
        point getB() { return B;}
        point getC() { return C;}

        void print() 
        {
            printf("face.A(%.2f,%.2f,%.2f),face.B(%.2f,%.2f,%.2f),face.C(%.2f,%.2f,%.2f)\n", A.getX(), A.getY(), A.getZ(), B.getX(), B.getY(), B.getZ(), C.getX(), C.getY(), C.getZ());
        }

        //得到三角面的两个边，以向量的形式返回
        dirVector getAB() { return dirVector(B.getX() - A.getX(), B.getY() - A.getY(), B.getZ() - A.getZ()); }
        dirVector getAC() { return dirVector(C.getX() - A.getX(), C.getY() - A.getY(), C.getZ() - C.getZ()); }

        void setA(point a) { A = a;} 
        void setB(point b) { B = b;} 
        void setC(point c) { C = c;} 

        //计算法向量
        dirVector getNormalVector()
        {
            float denominatorAB = sqrt((B.getX() - A.getX())*(B.getX() - A.getX()) + (B.getY() - A.getY())*(B.getY() - A.getY()) + (B.getZ() - A.getZ())*(B.getZ() - A.getZ()));
            float denominatorAC = sqrt((C.getX() - A.getX())*(C.getX() - A.getX()) + (C.getY() - A.getY())*(C.getY() - A.getY()) + (C.getZ() - A.getZ())*(C.getZ() - A.getZ()));
            dirVector AB = dirVector((B.getX() - A.getX())/denominatorAB, (B.getY() - A.getY())/denominatorAB, (B.getZ() - A.getZ())/denominatorAB);            
            dirVector AC = dirVector((C.getX() - A.getX())/denominatorAC, (C.getY() - A.getY())/denominatorAC, (C.getZ() - A.getZ())/denominatorAC);            
            return AB.chaCheng(AC);
        }
};

//射线类，包含一个原点和一个方向向量
class ray
{
    private:
        point origin;
        dirVector direction;
    public:
        ray(point p=point(0,0,0), dirVector dv=dirVector(1,0,0))
        {
            origin = p;
            direction = dv;
        }

        point getOrigin() { return origin; }
        dirVector getDirection() { return direction; }

        //打印出这条射线的信息，包括原点和方向向量
        void print()
        {
            printf("origin : x=%.2f y=%.2f z=%.2f ; direction : x=%.2f y=%.2f z=%.2f \n", origin.getX(), origin.getY(), origin.getZ(), direction.getX(), direction.getY(), direction.getZ());
        }

        void setOrigin(point p) { origin = p; }
        void setDirection(dirVector dv) { direction = dv; }

        //得到经过一个三角面反射的反射射线
        ray getReflectRayByFace(face reflectFace) 
        {
            dirVector normalVector = reflectFace.getNormalVector();//获取法向量

            float tmp1,tmp2,t;
            tmp1 = normalVector.dianCheng(dirVector(origin.getX(), origin.getY(), origin.getZ())) - normalVector.dianCheng(dirVector(reflectFace.getA().getX(),reflectFace.getA().getY(),reflectFace.getA().getZ()));
            tmp2 = normalVector.dianCheng(direction); 

            if (fabs(tmp2) < 0.000001)
            {
                return ray(point(0,0,0),dirVector(0,0,0));
            }
            else
            {
                t = -(tmp1/tmp2);
                point crossPoint = point(normalVector.getX()*t + origin.getX(), normalVector.getY()*t + origin.getY(), normalVector.getZ()*t + origin.getZ());
                point mirrorPoint = point(2*crossPoint.getX() - origin.getX(), 2*crossPoint.getY() - origin.getY(), 2*crossPoint.getZ() - origin.getZ());                

                if (crossPoint.isOnTheFace(reflectFace.getA(),reflectFace.getB(),reflectFace.getC())) 
                {
                    return ray(crossPoint, dirVector((mirrorPoint.getX() - crossPoint.getX()), (mirrorPoint.getY() - crossPoint.getY()), (mirrorPoint.getZ() - crossPoint.getZ())));
                }               
                else
                {
                    return ray(point(0,0,0),dirVector(0,0,0));
                }
            }
        }
};

//发射机类，包括一个发射机的点，以及天顶角(theta共180度)和方向角(phi共360度)两个球坐标方向各发射出多少条射线，所有的射线组成的数组
class transmitter
{
    private:
        point location;
        int thetaCnt, phiCnt;
        vector<ray> rays;
    public:
        transmitter( point p=point(0,0,0), int theta=180, int phi=360)
        {
            location = p;
            thetaCnt = theta;
            phiCnt = phi;
            for (int i = 0; i < thetaCnt; i ++)
            {
                for (int j = 0; j < phiCnt; j ++)
                {
                    dirVector dv = dirVector((float)(sin((float)i/(float)thetaCnt*(float)PI/2)*cos((float)j/(float)phiCnt*(float)PI)),(float)(sin((float)i/(float)thetaCnt*(float)PI/2)*sin((float)j/(float)phiCnt*(float)PI)),(float)(cos((float)i/(float)thetaCnt*(float)PI/2)));
                   // printf("x=%f,y=%f,z=%f\n", dv.getX(), dv.getY(), dv.getX());
                    ray tempRay = ray(location, dv);
                    rays.push_back(tempRay);
                }
            }
        }

        point getLocation() { return location; }
        int getThetaCnt() { return thetaCnt; }
        int getPhiCnt() { return phiCnt; }
        vector<ray> getRays() { return rays; }
        int getRaysCnt() { return (int)rays.size(); }

        void setLocation( point p ) { location = p; }
        void setThetaCnt( int theta ) { thetaCnt = theta; }
        void setPhiCnt( int phi ) { phiCnt = phi; }

        //让该发射机所有的射线遍历传进来的一个三角面的数组，返回所有的反射射线，没有反射射线的返回ray(point(0,0,0),dirVector(0,0,0))
        vector<ray> traverseFaces(vector<face>& faces)
        {
            long int tmpComputeCnt = 0;
            long int tmpSuccessCnt = 0;
            
            vector<ray> tmpRet;
            ray tmpRay;

            for ( vector<ray>::iterator itRay = rays.begin(); itRay < rays.end(); itRay ++ )
            {
                tmpComputeCnt = 0;
                tmpSuccessCnt = 0;
                for ( vector<face>::iterator itFace = faces.begin(); itFace < faces.end(); itFace ++ )
                {
                    tmpComputeCnt ++;
                    tmpRay = itRay->getReflectRayByFace(*itFace);
                    if (!(tmpRay.getOrigin().getX()==0 && tmpRay.getOrigin().getY()==0 && tmpRay.getOrigin().getZ()==0 && tmpRay.getDirection().getX()==0 && tmpRay.getDirection().getY()==0 && tmpRay.getDirection().getZ()==0))
                    {
                        tmpSuccessCnt ++;
                        tmpRet.push_back(tmpRay);
                    }
                }
                printf("tmpComputeCnt:%ld tmpSuccessCnt:%ld chazhi:%ld \n", tmpComputeCnt, tmpSuccessCnt, tmpComputeCnt - tmpSuccessCnt);
            }

            return tmpRet;
        }
};

template <int BLOCK_SIZE> __global__ void
rayReflectCUDA(ray *d_rays, face *d_faces, ray *d_retRay)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int faceBegin = bx * BLOCK_SIZE * BLOCK_SIZE;
    int faceEnd = (bx + 1) * BLOCK_SIZE * BLOCK_SIZE;
 
//    __shared__ face blockFaces[faceEnd - faceBegin];
    __shared__ face blockFaces[];
    int j = 0;
    for ( int i = faceBegin; i < faceEnd; i++ )
    {   
        blockFaces[j] = d_faces[i];
    }

    __syncthreads(); 
}


int rayReflect(int argc, char **argv)
{

    string pathTx = "./etc/tx.tx";string pathTer = "./etc/ter.ter";

    ifstream confTx(pathTx.c_str());
    ifstream confTer(pathTer.c_str());

    //读取发射机配置文件,得到一个point类型的对象
    transmitter tx;

    if (!confTx)
    {
        printf("open config file \" %s \" error \n",pathTx.c_str());
        exit(1);
    }
    else
    {
        string line;
        int position;
        bool flag = 0;

        while(getline(confTx,line))
        {
            position = line.find("nVertices");
            if (position != line.npos )
            {
                flag = 1;
                continue;
            }
            if (flag == 1)
            {
                float x = atof(line.substr(0,8).c_str());
                float y = atof(line.substr(9,7).c_str());
                float z = atof(line.substr(17,7).c_str());
                tx = transmitter(point(x,y,z),18,36);
                flag = 0;
            }
        }
        printf("########read tx config file OK######## \n\n");
        printf("    There are %d rays emit from one transmitter \n\n", tx.getRaysCnt());
    }
    //以上是读取发射机配置文件部分

    //读取地形三角面配置文件部分，返回一个face类型的数组
    vector<face> faces;    

    if (!confTer)
    {
        printf("open config file \" %s \" error \n",pathTer.c_str());
        exit(1);
    }
    else
    {
        string line;
        int position;
        int flag = 0;
        face tmpFace;

        while(getline(confTer, line))
        {
            position = line.find("nVertices");
            if (position != line.npos)
            {
                flag = 1;
                continue;
            }
            if (flag > 0)
            {
                if (flag == 1)
                {
                    float x,y,z;
                    vector<string> tmpRet;
                    tmpRet = split(line, " ");
                    int tmpCnt = 0;
                    for (vector<string>::iterator it = tmpRet.begin(); it < tmpRet.end(); it ++)
                    {
                        if (tmpCnt == 0)
                        {
                            x = atof(it->c_str());
                        }
                        if (tmpCnt == 1)
                        {
                            y = atof(it->c_str());
                        }
                        if (tmpCnt == 2)
                        {
                            z = atof(it->c_str());
                        }
                        tmpCnt ++;
                    }
                    tmpFace.setA(point(x,y,z));
                }
                if (flag == 2)
                {
                    float x,y,z;
                    vector<string> tmpRet;
                    tmpRet = split(line, " ");
                    int tmpCnt = 0;
                    for (vector<string>::iterator it = tmpRet.begin(); it < tmpRet.end(); it ++)
                    {
                        if (tmpCnt == 0)
                        {
                            x = atof(it->c_str());
                        }
                        if (tmpCnt == 1)
                        {
                            y = atof(it->c_str());
                        }
                        if (tmpCnt == 2)
                        {
                            z = atof(it->c_str());
                        }
                        tmpCnt ++;
                    }
                    tmpFace.setB(point(x,y,z));
                }
                if (flag == 3)
                {
                    float x,y,z;
                    vector<string> tmpRet;
                    tmpRet = split(line, " ");
                    int tmpCnt = 0;
                    for (vector<string>::iterator it = tmpRet.begin(); it < tmpRet.end(); it ++)
                    {
                        if (tmpCnt == 0)
                        {
                            x = atof(it->c_str());
                        }
                        if (tmpCnt == 1)
                        {
                            y = atof(it->c_str());
                        }
                        if (tmpCnt == 2)
                        {
                            z = atof(it->c_str());
                        }
                        tmpCnt ++;
                    }
                    tmpFace.setC(point(x,y,z));
                }
                
                flag ++;
                
                if (flag == 4) 
                { 
                    flag = 0;
                    faces.push_back(tmpFace);
                }
            }
        }

        printf("########read ter config file OK######## \n\n");
        printf("    There are %d triangle face on the map \n\n", (int)faces.size());
        
        printf("########compute begin########\n\n");

        clock_t start,end;
        start = clock();

        //Allocate host memory
        int raysCnt = tx.getRaysCnt();
        float h_rays[raysCnt*6];
        int i = 0;
        for (vector<ray>::iterator itRay = tx.getRays().begin(); itRay < tx.getRays().end(); itRay ++)
        {
            h_rays[i]   = *itRay->getOrigin()->getX();
            h_rays[i+1] = *itRay->getOrigin()->getY();
            h_rays[i+2] = *itRay->getOrigin()->getZ();

            h_rays[i+3] = *itRay->getDirection()->getX();
            h_rays[i+4] = *itRay->getDirection()->getY();
            h_rays[i+5] = *itRay->getDirection()->getZ();

            i += 6;
        }

        int facesCnt = (int)faces.size();
        float h_faces[facesCnt*9];
        i = 0;
        for (vector<face>::iterator itFace = faces.begin(); itFace < faces.end(); itFace ++)
        {
            h_faces[i]   = *itFace->getA()->getX();
            h_faces[i+1] = *itFace->getA()->getY();
            h_faces[i+2] = *itFace->getA()->getZ();

            h_faces[i+3] = *itFace->getB()->getX();
            h_faces[i+4] = *itFace->getB()->getY();
            h_faces[i+5] = *itFace->getB()->getZ();

            h_faces[i+6] = *itFace->getC()->getX();
            h_faces[i+7] = *itFace->getC()->getY();
            h_faces[i+8] = *itFace->getC()->getZ();

            i += 9;
        }

        unsigned int mem_size_rays = (unsigned int)(sizeof(h_rays);
        unsigned int mem_size_faces = (unsigned int)(sizeof(h_faces);

        float *h_resultRays = (float *) malloc(mem_size_rays);

        if (h_resultRays == NULL)
        {
            fprintf(stderr, "Failed to allocate host vector resultRays ! \n");
            exit(EXIT_FAILURE);
        }

        //Allocate device memory
        float *d_rays;
        float *d_faces;       
        float *d_resultRays;       
  
        cudaError_t error;
        error = cudaMalloc((void **) &d_rays, mem_size_rays);
        
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_rays returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

        error = cudaMalloc((void **) &d_faces, mem_size_faces);

        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_faces returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

        error = cudaMalloc((void **) &d_resultRays, mem_size_rays);

        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_resultRays returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

        //copy host memory to device
        error = cudaMemcpy(d_rays, h_rays, mem_size_rays, cudaMemcpyHostToDevice);

        if (error != cudaSuccess)
        {
            printf("cudaMemcpy (d_rays, h_rays) returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE); 
        }

        error = cudaMemcpy(d_faces, h_faces, mem_size_faces, cudaMemcpyHostToDevice);

        if (error != cudaSuccess)
        {
            printf("cudaMemcpy (d_faces, h_faces) returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

        //Setup execution parameters
        int block_size = 32;

        dim3 threads(block_size, block_size);
        dim3 grid(gridX, gridY);

        if (block_size == 16)
        {
            rayReflectCUDA<16><<< grid, threads >>>(d_rays, d_faces, d_resultRays);
        }
        else
        {
            rayReflectCUDA<32><<< grid, threads >>>(d_rays, d_faces, d_resultRays);
        }
        printf("    done\n\n");

        end = clock();
        float useTime = (end - start)/1000;
        printf("    use %.2f (ms)\n\n", useTime);

        free(h_rays);
        free(h_faces);
        free(h_resultRays);
        cudaFree(d_rays);
        cudaFree(d_faces);
        cudaFree(d_resultRays);

    }

    confTx.close();
    confTer.close();

    return 0;
}

int main(int argc, char **argv)
{
    printf("########[Ray Reflect Using CUDA] - Starting...########\n\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -setup=setup.setup the setup config file in the direction of ./etc/\n");
        printf("      -tx=tx.tx the transmitter config file in the direction of ./etc\n");
        printf("      -rx=rx.rx the receiver config file in the direction of ./etc\n");
        printf("      -ter=ter.ter the map config file in the direction of ./etc\n");

        exit(EXIT_SUCCESS);
    }

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        cudaSetDevice(devID);
    }

    cudaError_t error;
    cudaDeviceProp deviceProp;                                                                                                                                     
    error = cudaGetDevice(&devID);                                                                                                                                 
                                                                                                                                                                   
    if (error != cudaSuccess)                                                                                                                                      
    {                                                                                                                                                              
        printf("cudaGetDevice returned error code %d, line(%d)\n\n", error, __LINE__);                                                                               
    }                                                                                                                                                              
                                                                                                                                                                   
    error = cudaGetDeviceProperties(&deviceProp, devID);                                                                                                           
                                                                                                                                                                   
    if (deviceProp.computeMode == cudaComputeModeProhibited)                                                                                                       
    {                                                                                                                                                              
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");                                         
        exit(EXIT_SUCCESS);                                                                                                                                        
    }                                                                                                                                                              
                                                                                                                                                                   
    if (error != cudaSuccess)                                                                                                                                      
    {                                                                                                                                                              
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);                                                                     
    }      
    else                                                                                                                                                           
    {                                                                                                                                                              
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);                             
    }                                                                                                                                                              
                                                                                                                                                                   
   int result = rayReflect(argc,argv);
                                                                                                                                                               
    exit(result);
}
