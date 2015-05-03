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
        dirVector(float x=0, float y=0, float z=0)
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
        dirVector chaCheng(dirVector tmp)
        {
            return dirVector((Y*tmp.Z - Z*tmp.Y), (Z*tmp.X - X*tmp.Z), (X*tmp.Y - Y*tmp.X));
        }
        //计算两个向量点乘
        float dianCheng(dirVector tmp)
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
        point(float x=0, float y=0, float z=0)
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

struct d_point
{
    float X;
    float Y;
    float Z;
};

struct d_dirVector
{
    float X;
    float Y;
    float Z;
};

struct d_ray
{
    struct d_point origin;
    struct d_dirVector direction;
};

struct d_face
{
    float AX;
    float AY;
    float AZ;
    float BX;
    float BY;
    float BZ;
    float CX;
    float CY;
    float CZ;
};

__device__ float d_pointDianCheng(struct d_point A, struct d_point B)
{
    return A.X * B.X + A.Y * B.Y + A.Z * B.Z;
};

__device__ float d_dirVectorDianCheng(struct d_dirVector A, struct d_dirVector B)
{
    return A.X * B.X + A.Y * B.Y + A.Z * B.Z;
};

__device__ float d_pointVectorDianCheng(struct d_point A, struct d_dirVector B)
{
    return A.X * B.X + A.Y * B.Y + A.Z * B.Z;
};

__device__ struct d_point d_pointChaCheng(struct d_point A, struct d_point B)
{
    struct d_point result = {A.Y * B.Z - B.Y * A.Z , A.Z * B.X - A.X * B.Z , A.X * B.Y - A.Y * B.X};
    return result;
};

__device__ struct d_dirVector d_dirVectorChaCheng(struct d_dirVector A, struct d_dirVector B)
{
    struct d_dirVector result = {A.Y * B.Z - B.Y * A.Z , A.Z * B.X - A.X * B.Z , A.X * B.Y - A.Y * B.X};
    return result;
};

__device__ float mag(d_dirVector vector)
{
    return sqrt(vector.X * vector.X + vector.Y * vector.Y + vector.Z * vector.Z);
}

__device__ d_dirVector getReverseVector(d_dirVector vector)
{
    d_dirVector retVector = {-vector.X , -vector.Y , -vector.Z};
    return retVector;
}

__device__ float getPhaseOfVector(d_dirVector firstVector , d_dirVector secondVector)
{
    float multipliedTemp, nTemp;
    multipliedTemp = d_dirVectorDianCheng(firstVector , secondVector);
    float tempValue = sqrt(pow(firstVector.X, 2) + pow(firstVector.Y, 2) + pow(firstVector.Z, 2)) * sqrt(pow(secondVector.X, 2) + pow(secondVector.Y, 2) + pow(secondVector.Z, 2));

    if (tempValue < 0.001)
    {
        nTemp = 0;
    }
    else
    {
        nTemp = multipliedTemp / tempValue;
    }
    float phase = acos(nTemp);
    return phase;
}

__device__ d_dirVector getRightRotationVector(d_dirVector thisVector , d_dirVector vector, float rotationAngle)
{
    float normalizations[4];
    float rotateparameters[9];
    d_dirVector rotationVector = {vector.X / mag(vector) , vector.Y / mag(vector) , vector.Z / mag(vector)};
    normalizations[0] = rotationVector.X;
    normalizations[1] = rotationVector.Y;
    normalizations[2] = rotationVector.Z;
    normalizations[3] = rotationAngle;

    rotateparameters[0] = cos(normalizations[3]) + normalizations[0] * normalizations[0] * (1 - cos(normalizations[3]));
    rotateparameters[1] = normalizations[0] * normalizations[1] * (1 - cos(normalizations[3])) + normalizations[2] * sin(normalizations[3]);
    rotateparameters[2] = normalizations[0] * normalizations[2] * (1 - cos(normalizations[3])) - normalizations[1] * sin(normalizations[3]);
    rotateparameters[3] = normalizations[0] * normalizations[1] * (1 - cos(normalizations[3])) - normalizations[2] * sin(normalizations[3]);
    rotateparameters[4] = cos(normalizations[3]) + normalizations[1] * normalizations[1] * (1 - cos(normalizations[3]));
    rotateparameters[5] = normalizations[1] * normalizations[2] * (1 - cos(normalizations[3])) + normalizations[0] * sin(normalizations[3]);
    rotateparameters[6] = normalizations[0] * normalizations[2] * (1 - cos(normalizations[3])) + normalizations[1] * sin(normalizations[3]);
    rotateparameters[7] = normalizations[1] * normalizations[2] * (1 - cos(normalizations[3])) - normalizations[0] * sin(normalizations[3]);
    rotateparameters[8] = cos(normalizations[3]) + normalizations[2] * normalizations[2] * (1 - cos(normalizations[3]));
    float xtemp = thisVector.X * rotateparameters[0] + thisVector.Y * rotateparameters[3] + thisVector.Z * rotateparameters[6];
    float ytemp = thisVector.X * rotateparameters[1] + thisVector.Y * rotateparameters[4] + thisVector.Z * rotateparameters[7];
    float ztemp = thisVector.X * rotateparameters[2] + thisVector.Y * rotateparameters[5] + thisVector.Z * rotateparameters[8];
    d_dirVector retDirVector = {xtemp , ytemp , ztemp};
    return retDirVector;
}

__device__ struct d_dirVector d_getNormalVector(struct d_face face)
{
   struct d_point origin = {face.AX , face.AY , face.AZ};
   struct d_dirVector direction = {face.BX - face.AX , face.BY - face.AY , face.BZ - face.AZ};
   struct d_ray AB = {origin , direction }; 

   struct d_dirVector direction1 = {face.CX - face.AX , face.CY - face.AY , face.CZ - face.AZ};
   struct d_ray AC = {origin , direction1}; 

   struct d_dirVector normalVector = d_dirVectorChaCheng(AB.direction , AC.direction);
   float denominator = sqrt(d_dirVectorDianCheng(AB.direction,AB.direction) + d_dirVectorDianCheng(AC.direction,AC.direction));
   if (denominator < 0.001)
   {
       struct d_dirVector normalVectorGuiyi = {0 , 0 , 0};
       return normalVectorGuiyi;
   }
   else
   {
       struct d_dirVector normalVectorGuiyi = {normalVector.X / denominator, normalVector.Y / denominator, normalVector.Z / denominator};
       return normalVectorGuiyi;
   }
}

__device__ bool pointIsOnFace(struct d_point origin, struct d_face face)
{
    if ( (origin.X == face.AX && origin.Y == face.AY && origin.Z == face.AZ) || (origin.X == face.BX && origin.Y == face.BY && origin.Z == face.BZ) || (origin.X == face.CX && origin.Y == face.CY && origin.Z == face.CZ) )     
    {                                                                                                                                                      
        return true;                                                                                                                                       
    }                                                                                                                                                      
    else                                                                                                                                                   
    {                                                                                                                                                      
        struct d_dirVector ab = {face.BX - face.AX, face.BY - face.AY, face.BZ - face.AZ};                                                                     
        struct d_dirVector ad = {origin.X - face.AX, origin.Y - face.AY, origin.Z - face.AZ}; 
        struct d_dirVector bc = {face.CX - face.BX, face.CY - face.BY, face.CZ - face.BZ};                                                                     
        struct d_dirVector bd = {origin.X - face.BX, origin.Y - face.BY, origin.Z - face.BZ};
        struct d_dirVector cd = {origin.X - face.CX, origin.Y - face.CY, origin.Z - face.CZ};
        struct d_dirVector ca = {face.AX - face.CX, face.AY - face.CY, face.AZ - face.CZ};                                                                     
                                                                                                                                                           
        d_dirVector adXab = d_dirVectorChaCheng(ad,ab);
        d_dirVector bdXbc = d_dirVectorChaCheng(bd,bc);
        d_dirVector cdXca = d_dirVectorChaCheng(cd,ca);
                                                                                                                                                           
        float m,n,l,p,q;                                                                                                                                   
        m = d_dirVectorDianCheng(adXab,bdXbc);                                                                                                                        
        n = d_dirVectorDianCheng(bdXbc,cdXca);                                                                                                                        
        l = d_dirVectorDianCheng(cdXca,adXab);                                                                                                                        
                                                                                                                                                           
        q = d_dirVectorDianCheng(adXab,ab);                                                                                                                           
        p = d_dirVectorDianCheng(adXab,bc);                                                                                                                           
                                                                                                                                                           
        if (m >= 0 && n >= 0 && l >= 0 && p == 0 && q == 0) { return true; }                                                                               
        else { return false; }                                                                                                                             
    }                                                                                                                                                      
}

    //得到经过一个三角面反射的反射射线
__device__ struct d_ray getReflectRayByFace(d_ray ray, d_face face) 
{
    struct d_dirVector normalVector = d_getNormalVector(face);//获取法向量

    float tmp1,tmp2,t;
    struct d_point faceA = {face.AX , face.AY , face.AZ};
    tmp1 = d_pointVectorDianCheng(ray.origin, normalVector) - d_pointVectorDianCheng(faceA, normalVector);
    tmp2 = d_dirVectorDianCheng(normalVector , ray.direction); 

    if (fabs(tmp2) < 0.0001)
    {
        struct d_point resultPoint = {0 , 0 , 0};
        struct d_dirVector resultDirVector = {0 , 0 , 0};
        struct d_ray result = {resultPoint , resultDirVector};
        return result;
    }
    else
    {
        t = -(tmp1/tmp2);
        struct d_point crossPoint = {normalVector.X * t + ray.origin.X , normalVector.Y * t + ray.origin.Y , normalVector.Z * t + ray.origin.Z};
        struct d_point mirrorPoint = {2*crossPoint.X - ray.origin.X, 2*crossPoint.Y - ray.origin.Y, 2*crossPoint.Z - ray.origin.Z};                

        if (pointIsOnFace(crossPoint , face)) 
        {
            struct d_dirVector resultDir= {mirrorPoint.X - crossPoint.X , mirrorPoint.Y - crossPoint.Y , mirrorPoint.Z - crossPoint.Z};
            struct d_ray result = {crossPoint, resultDir};
            return result;
        }               
        else
        {
            struct d_point resultPoint = {0 , 0 , 0};
            struct d_dirVector resultDirVector = {0 , 0 , 0};
            struct d_ray result = {resultPoint , resultDirVector};
            return result;
        }
    }
};

__device__ d_point getIntersectionPointWithFace(d_ray ray , d_face face)
{   
    d_dirVector normalVector = d_getNormalVector(face);

    float temp = fabs(d_dirVectorDianCheng(ray.direction , normalVector));
    if (temp < 0.001)
    {
        d_point retPoint = {0 , 0 , 0};
        return retPoint;
    }
    else
    {   
        float TriD = normalVector.X * face.AX+ normalVector.Y * face.BY + normalVector.Z * face.CZ;
        float temp1 = normalVector.X * ray.origin.X+ normalVector.Y * ray.origin.Y + normalVector.Z * ray.origin.Z - TriD;
        float t = -temp1 / temp;
        d_point tempPoint = {ray.direction.X * t + ray.origin.X, ray.direction.Y * t + ray.origin.Y, ray.direction.Z * t + ray.origin.Z};
        if (pointIsOnFace(tempPoint , face))
        {
            return tempPoint;
        }
        else
        {
            tempPoint.X = 0;
            tempPoint.Y = 0; 
            tempPoint.Z = 0;
            return tempPoint;
        }   
    }   
};

__device__ d_ray getRayOut(d_ray ray , d_face face)
{   
    d_dirVector normalVector = d_getNormalVector(face);

    if ((mag(ray.direction) * mag(normalVector)) < 0.0001)
    {
        d_point retPoint = {0 , 0 , 0};
        d_dirVector retVector = {0 , 0 , 0};
        d_ray retRay = {retPoint , retVector};
        return retRay;
    }
    else
    {
        float temp=(d_dirVectorDianCheng(ray.direction , normalVector) / (mag(ray.direction) * mag(normalVector)));

        if (fabs(temp-1)<0.001)
        {
            d_ray retRay = {getIntersectionPointWithFace(ray , face), getReverseVector(ray.direction)};
            return retRay;
        }
       
        if (fabs(d_dirVectorDianCheng(ray.direction , normalVector)) < 0.001)
        {
            d_point retPoint = {0 , 0 , 0};
            d_dirVector retVector = {0 , 0 , 0};
            d_ray retRay = {retPoint , retVector};
            return retRay;
        }
    
        d_dirVector rotationVector = getReverseVector(d_dirVectorChaCheng(ray.direction , normalVector));
        float reflectAngle = getPhaseOfVector(ray.direction , normalVector);
        float rotationAngle = PI - 2 * reflectAngle;
        d_dirVector vectorOfRfraction = getRightRotationVector(ray.direction , rotationVector, rotationAngle);
        d_point crosspoint = getIntersectionPointWithFace(ray , face);
        d_ray retRay = {crosspoint , vectorOfRfraction};
        return retRay;
    }
}

template <int BLOCK_SIZE, int BLOCK_SIZE_FACE> __global__ void
rayReflectCUDA(int rays_size, int faces_size, float d_rays[], float d_faces[], float d_retRay[])
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int GRID_SIZE = 4;

    //每个block的shared memory存储的三角面的长度
    int faceStep = (int)(BLOCK_SIZE_FACE * BLOCK_SIZE_FACE * 9);

    //每个thread所计算的ray在整个d_rays中的偏移
    int shift = (GRID_SIZE * by + bx) * BLOCK_SIZE * BLOCK_SIZE + ty * BLOCK_SIZE + tx;

    //提取每个thread需要计算的射线
    struct d_point ray_point = {d_rays[shift] , d_rays[shift + 1] , d_rays[shift + 2]};
    struct d_dirVector ray_dirVector = {d_rays[shift + 3] , d_rays[shift + 4] , d_rays[shift + 5]};
    struct d_ray each_ray = {ray_point , ray_dirVector};

    for ( int i = 0; i < faces_size; i += faceStep )
    {   
        __shared__ float blockFaces[(int)(BLOCK_SIZE_FACE * BLOCK_SIZE_FACE * 9)];

        if ((ty%2==0) && (tx%2==0))
        {
            for ( int j = 0; j < 9; j++ )
            {
                blockFaces[(BLOCK_SIZE_FACE * ty / 2  + tx / 2) * 9 + j] = d_faces[i + (BLOCK_SIZE_FACE * ty / 2 + tx / 2) * 9 + j];
            }
        }

        __syncthreads(); 

        for ( int i = 0; i < BLOCK_SIZE_FACE*BLOCK_SIZE_FACE*9; i += 9)
        {
            struct d_face each_face = {blockFaces[i] , blockFaces[i + 1] , blockFaces[i + 2] , blockFaces[i + 3] , blockFaces[i + 4] , blockFaces[i + 5] , blockFaces[i + 6] , blockFaces[i + 7] , blockFaces[i + 8]};

//            struct d_ray result = getReflectRayByFace(each_ray, each_face);
            d_ray result = getRayOut(each_ray , each_face);           
          
            d_retRay[shift] = result.origin.X;
            d_retRay[shift+1] = result.origin.Y;
            d_retRay[shift+2] = result.origin.Z;
            d_retRay[shift+3] = result.direction.X;
            d_retRay[shift+4] = result.direction.Y;
            d_retRay[shift+5] = result.direction.Z;
        }
       
        __syncthreads();

    }
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
                tx = transmitter(point(x,y,z),256,256);
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

        int raysCnt = tx.getRaysCnt();
        int facesCnt = (int)faces.size();

        unsigned int mem_size_rays = (raysCnt * 6) * sizeof(float);
        unsigned int mem_size_faces = (facesCnt * 9) * sizeof(float);

        float *h_rays = (float *)malloc(mem_size_rays);

        if (h_rays == NULL)
        {
            fprintf(stderr, "Failed to allocate host vector h_rays ! \n");
            exit(EXIT_FAILURE);
        }

        float *h_faces = (float *)malloc(mem_size_faces);

        if (h_faces == NULL)
        {
            fprintf(stderr, "Failed to allocate host vector h_faces ! \n");
            exit(EXIT_FAILURE);
        }

        //Allocate host memory
        int i = 0;
        vector<ray> rrrays = tx.getRays();
        for (vector<ray>::iterator itRay = rrrays.begin(); itRay < rrrays.end(); itRay ++)
        {
            h_rays[i]   = itRay->getOrigin().getX();
            h_rays[i+1] = itRay->getOrigin().getY();
            h_rays[i+2] = itRay->getOrigin().getZ();

            h_rays[i+3] = itRay->getDirection().getX();
            h_rays[i+4] = itRay->getDirection().getY();
            h_rays[i+5] = itRay->getDirection().getZ();

            i += 6;
        }

        i = 0;
        for (vector<face>::iterator itFace = faces.begin(); itFace < faces.end(); itFace ++)
        {
            h_faces[i]   = itFace->getA().getX();
            h_faces[i+1] = itFace->getA().getY();
            h_faces[i+2] = itFace->getA().getZ();

            h_faces[i+3] = itFace->getB().getX();
            h_faces[i+4] = itFace->getB().getY();
            h_faces[i+5] = itFace->getB().getZ();

            h_faces[i+6] = itFace->getC().getX();
            h_faces[i+7] = itFace->getC().getY();
            h_faces[i+8] = itFace->getC().getZ();

            i += 9;
        }

        float *h_resultRays = (float *)malloc(mem_size_rays);

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
        int gridX = 4;
        int gridY = 4;

        dim3 threads(block_size, block_size);
        dim3 grid(gridX, gridY);

        // Allocate CUDA events that we'll use for timing
        cudaEvent_t start;
        error = cudaEventCreate(&start);
    
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    
        cudaEvent_t stop;
        error = cudaEventCreate(&stop);
    
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);                                                                                                                                        
        }                                                                                                                                                              
                                                                                                                                                                       
        // Record the start event                                                                                                                                      
        error = cudaEventRecord(start, NULL);                                                                                                                          
                                                                                                                                                                       
        if (error != cudaSuccess)                                                                                                                                      
        {                                                                                                                                                              
            fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));                                                             
            exit(EXIT_FAILURE);                                                                                                                                        
        }

        ///////////////开始进入kernel计算
        rayReflectCUDA<32,16><<< grid, threads >>>(raysCnt*6, facesCnt*9, d_rays, d_faces, d_resultRays);

        cudaDeviceSynchronize();

        // Record the stop event                                                                                                                                       
        error = cudaEventRecord(stop, NULL);                                                                                                                           
                                                                                                                                                                       
        if (error != cudaSuccess)                                                                                                                                      
        {                                                                                                                                                              
            fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));                                                              
            exit(EXIT_FAILURE);                                                                                                                                        
        }                                                                                                                                                              
                                                                                                                                                                       
        // Wait for the stop event to complete                                                                                                                         
        error = cudaEventSynchronize(stop);                                                                                                                            
                                                                                                                                                                       
        if (error != cudaSuccess)                                                                                                                                      
        {                                                                                                                                                              
            fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));                                                  
            exit(EXIT_FAILURE);                                                                                                                                        
        }                                                                                                                                                              
                                                                                                                                                                       
        float msecTotal = 0.0f;                                                                                                                                        
        error = cudaEventElapsedTime(&msecTotal, start, stop);                                                                                                         
                                                                                                                                                                       
        if (error != cudaSuccess)                                                                                                                                      
        {                                                                                                                                                              
            fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));                                                
            exit(EXIT_FAILURE);                                                                                                                                        
        }                                                                                                                                                              
                                                                                                                                                                       
        // Compute and print the performance                                                                                                                           
        printf("        use %.3f (ms) \n\n", msecTotal);

        error = cudaMemcpy(h_resultRays, d_resultRays, mem_size_rays, cudaMemcpyDeviceToHost);

        if (error != cudaSuccess)
        {
            printf("cudaMemcpy (h_resultRays, d_resultRays) returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

        for (int i = 0 ; i < mem_size_rays / sizeof(float); i += 6)
        {
            printf("   point: %.2f %.2f %.2f ;  %.2f %.2f %.2f \n", h_resultRays[i] , h_resultRays[i+1] , h_resultRays[i+2] , h_resultRays[i+3] , h_resultRays[i+4] , h_resultRays[i+5]);
        }

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
