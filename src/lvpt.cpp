// Lite Kevin Beason

#define _USE_MATH_DEFINES


#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
#include <algorithm>
#pragma warning(disable: 4244) // Disable double to float warning

// Usage: time ./smallpt 5000 && xv image.ppm


// Material types used in computing the radiance (L) values
enum ReflectionType { DIFF, SPEC, REFR };

// XOR shift PRNG
namespace XORShift
{
unsigned int x = 123456789;
unsigned int y = 362436069;
unsigned int z = 521288629;
unsigned int w = 88675123;

inline float RNG()
{
    unsigned int t;
    t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    return ( w = ( w ^ ( w >> 19 )) ^ ( t ^ ( t >> 8 ))) * ( 1.0f / 4294967295.0f );
}
}

// This structure can be used for XYZ position or RGB color
struct Vector3
{
    // Data
    double x;
    double y;
    double z;

    // Constructor
    Vector3( const double X = 0, const double Y = 0, const double Z = 0 )
    {
        x = X;
        y = Y;
        z = Z;
    }

    // Computes the length of the vector in the 3D space
    float Length( )
    {
        return sqrt(( x * x ) + ( y * y ) + ( z * z ));
    }

    // Computes the dot product between this vector and another vector _vec_
    double DotProduct( const Vector3 &vec ) const
    {
        return (( x * vec.x ) + ( y * vec.y ) + ( z * vec.z ));
    }

    // Computes the cross product between this vector and another vector _vec_
    double CrossProduct( const Vector3 vec )
    {

    }

    // Operators
    Vector3 operator+( const Vector3 &vec ) const
    {
        return Vector3(( x + vec.x ), ( y + vec.y ), ( z + vec.z ));
    }

    Vector3 operator-( const Vector3 &vec ) const
    {
        return Vector3(( x - vec.x ), ( y - vec.y ), ( z - vec.z ));
    }

    Vector3 operator*( double constant ) const
    {
        return Vector3( x * constant, y * constant, z * constant );
    }

    Vector3 operator%( Vector3&vec )
    {
        return Vector3(( y * vec.z ) - ( z * vec.y ),
                       ( z * vec.x ) - ( x * vec.z ),
                       ( x * vec.y ) - ( y * vec.x ));
    }

    Vector3 Multiply( const Vector3 &vec ) const
    {
        return Vector3(x*vec.x,y*vec.y,z*vec.z);
    }

    Vector3& Normalize( )
    {
        return *this = *this * (1/sqrt(x*x+y*y+z*z));
    }
};

typedef Vector3 Color3;
typedef Vector3 Emission3;


// Ray structure for origin and direction
struct Ray
{
    Vector3 origin;
    Vector3 direction;

    // Default constructor
    Ray( ) { }

    // Constructor initialized with origin and direction
    Ray( Vector3 initOrigin, Vector3 initDirection ) :
        origin( initOrigin ),
        direction( initDirection ) { }
};

// Sphere geometry
struct Sphere
{
    // Sphere data
    double          radius;
    Vector3         position;
    Vector3         emission;
    Vector3         color;
    ReflectionType  reflection;

    // Constructor
    Sphere( double initRadius, Vector3 initPos, Vector3 initEmission,
            Vector3 initColor, ReflectionType initReflection ) :
        radius( initRadius ),
        position( initPos ),
        emission( initEmission ),
        color( initColor ),
        reflection( initReflection ) { }

    // Testing intersection with the _ray_ and return intersection point _tIntersection_
    // Returns distance, and 0 if it doesn't hit with any thing
    double Intersect( const Ray &ray, double *tIntersection = NULL,
                      double *tOut = NULL ) const
    {
        Vector3 op = position-ray.origin; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t;
        double eps=1e-4;
        double b=op.DotProduct(ray.direction);
        double det=b*b-op.DotProduct(op)+radius*radius;

        if( det < 0 )
            return 0;
        else
            det = sqrt( det );

        if ( tIntersection && tOut )
        {
            *tIntersection = (b - det <= 0) ? 0:b - det;
            *tOut = b + det;
        }

        return ( t=b-det ) > eps ? t : (( t = b + det) > eps ? t : 0 );
    }
};


inline double Clamp( const double value )
{
    return value < 0 ? 0 : value > 1 ? 1 : value;
}

inline int ColorToInteger( const double value )
{
    return int( pow( Clamp( value ), 1 / 2.2) * 255 + 0.5);
}

// Creating a scene that is only composed of spheres.
Sphere sceneSpheres[] =
{
    Sphere( 25.0, Vector3( 27.0, 18.5, 78.0 ), Emission3( 00.0, 00.0, 00.0 ),
    Color3( 1.00, 1.00, 1.00 ) * 0.75, SPEC ),    //Mirr
    Sphere( 12.0, Vector3( 70.0, 43.0, 78.0 ), Emission3( 00.0, 00.0, 00.0 ),
    Color3( 0.27, 0.80, 0.80 ) * 1.00, REFR ),    //Glas
    Sphere( 8.00, Vector3( 55.0, 87.0, 78.0 ), Emission3( 00.0, 00.0, 00.0 ),
    Color3( 1.00, 1.00, 1.00 ) * 0.75, DIFF ),    //Lite
    Sphere( 4.00, Vector3( 55.0, 80.0, 78.0 ), Emission3( 10.0, 10.0, 10.0 ),
    Color3( 00.0, 00.0, 00.0 ) * 1.00, DIFF )     //Lite
};

// Homogenous sphere (participating media)
Sphere homogenousSphere(300, Vector3(50,50,80), Vector3(), Vector3(), DIFF);

// Optical properties of the volume
const double sigma_s = 0.009;
const double sigma_a = 0.006;
const double sigma_t = sigma_s + sigma_a;


// Test intersection between ray and sphere
inline bool Intersects( const Ray &ray,
                        double &tIntsct,
                        int &objId,
                        const double tMax = 1e20 )
{
    // Number of spheres in the scene
    double numSpheres = sizeof(sceneSpheres) / sizeof(Sphere);
    double itsctDistance;
    double INFINITY_DISTANCE = tIntsct = tMax;


    // Checks if the ray intersects any sphere or not.
    for(int iSphere = int( numSpheres ); iSphere--;)
    {
        if(( itsctDistance = sceneSpheres[iSphere].Intersect( ray ))
                && itsctDistance < tIntsct )
        {
            tIntsct = itsctDistance;
            objId = iSphere;
        }
    }

    return tIntsct < INFINITY_DISTANCE;
}

inline double SampleSegment( const double epsilon,
                             const float sigma,
                             const float sMax )
{
    return -log( 1.0 - epsilon * ( 1.0 - exp( -sigma * sMax ))) / sigma;
}

// Uniform phase function g = 0, by sampling a sphere
inline Vector3 SampleSphere(double e1, double e2)
{
    double z = ( 1.0 - ( 2.0 * e1 ));
    double sint = sqrt( 1.0 - ( z * z ));

    return Vector3( cos( 2.0 * M_PI * e2 ) * sint,
                    sin( 2.0 * M_PI * e2 ) * sint,
                    z);
}


// HG Phase function
inline Vector3 SampleHG(double g, double e1, double e2) {
    //double s=2.0*e1-1.0, f = (1.0-g*g)/(1.0+g*s), cost = 0.5*(1.0/g)*(1.0+g*g-f*f), sint = sqrt(1.0-cost*cost);

    double s = 1.0 - 2.0 * e1;
    double cost = (s + 2.0 * g * g * g * (-1.0 + e1) * e1 + g * g * s + 2.0 * g *(1.0 - e1 + e1 * e1)) / (( 1.0 + g * s) * (1.0 + g * s));
    double sint = sqrt( 1.0 - cost * cost);

    return Vector3( cos( 2.0 * M_PI * e2 ) * sint,
                    sin( 2.0 * M_PI * e2 ) * sint,
                    cost);
}
inline void GenerateOrthoBasis(Vector3 &u, Vector3 &v, Vector3 w) {
    Vector3 coVec = w;
    if (fabs(w.x) <= fabs(w.y))
        if (fabs(w.x) <= fabs(w.z)) coVec = Vector3(0,-w.z,w.y);
        else coVec = Vector3(-w.y,w.x,0);
    else if (fabs(w.y) <= fabs(w.z)) coVec = Vector3(-w.z,0,w.x);
    else coVec = Vector3(-w.y,w.x,0);
    coVec.Normalize();
    u = w%coVec,
            v = w%u;
}

// Internal scattering in the volume
inline double Scatter(const Ray &ray, Ray *sRay, double tIntsct, float tOut, double &segment)
{
    // ?
    segment = SampleSegment(XORShift::RNG(), sigma_s, tOut - tIntsct);

    Vector3 x = ray.origin + ray.direction * tIntsct + ray.direction * segment;

    // Sample a direction ~ uniform phase function
    // Vector3 direction = SampleSphere(XORShift::frand( ), XORShift::frand( ));

    // Sample a direction ~ HG phase function
    Vector3 direction = SampleHG(-0.5, XORShift::RNG( ), XORShift::RNG( ));

    // Parameteric variable for determining the intersection point
    Vector3 u,v;

    GenerateOrthoBasis( u, v, ray.direction );

    direction = ( u * direction.x ) +
            ( v * direction.y ) +
            ( ray.direction * direction.z );
    if (sRay)
        *sRay = Ray( x, direction );
    return ( 1.0 - exp( -sigma_s * ( tOut - tIntsct )));
}

Vector3 LRadiance(const Ray &r, int depth) {
    double t;                               // distance to intersection
    int id=0;                               // id of intersected object
    double tnear, tfar, scaleBy=1.0, absorption=1.0;
    bool intrsctmd = homogenousSphere.Intersect(r, &tnear, &tfar) > 0;
    if (intrsctmd) {
        Ray sRay;
        double s, ms = Scatter(r, &sRay, tnear, tfar, s), prob_s = ms;
        scaleBy = 1.0/(1.0-prob_s);
        if (XORShift::RNG() <= prob_s) {// Sample surface or volume?
            if (!Intersects(r, t, id, tnear + s))
                return LRadiance(sRay, ++depth) * ms * (1.0/prob_s);
            scaleBy = 1.0;
        }
        else
            if (!Intersects(r, t, id)) return Vector3();
        if (t >= tnear) {
            double dist = (t > tfar ? tfar - tnear : t - tnear);
            absorption=exp(-sigma_t * dist);
        }
    }
    else
        if (!Intersects(r, t, id)) return Vector3();
    const Sphere &obj = sceneSpheres[id];        // the hit object
    Vector3 x=r.origin+r.direction*t, n=(x-obj.position).Normalize(), nl=n.DotProduct(r.direction)<0?n:n*-1, f=obj.color,Le=obj.emission;
    double p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; // max refl
    if (++depth>5) if (XORShift::RNG()<p) {f=f*(1/p);} else return Vector3(); //R.R.
    if (n.DotProduct(nl)>0 || obj.reflection != REFR) {f = f * absorption; Le = obj.emission * absorption;}// no absorption inside glass
    else scaleBy=1.0;
    if (obj.reflection == DIFF) {                  // Ideal DIFFUSE reflection
        double r1=2*M_PI*XORShift::RNG(), r2=XORShift::RNG(), r2s=sqrt(r2);
        Vector3 w=nl, u=((fabs(w.x)>.1?Vector3(0,1):Vector3(1))%w).Normalize(), v=w%u;
        Vector3 d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).Normalize();
        return (Le + f.Multiply(LRadiance(Ray(x,d),depth))) * scaleBy;
    } else if (obj.reflection == SPEC)            // Ideal SPECULAR reflection
        return (Le + f.Multiply(LRadiance(Ray(x,r.direction-n*2*n.DotProduct(r.direction)),depth))) * scaleBy;
    Ray reflRay(x, r.direction-n*2*n.DotProduct(r.direction));     // Ideal dielectric REFRACTION
    bool into = n.DotProduct(nl)>0;                // Ray from outside going in?
    double nc=1, nt=1.5, nnt=into?nc/nt:nt/nc, ddn=r.direction.DotProduct(nl), cos2t;
    if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0)    // Total internal reflection
        return (Le + f.Multiply(LRadiance(reflRay,depth)));
    Vector3 tdir = (r.direction*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).Normalize();
    double a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:tdir.DotProduct(n));
    double Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP=Tr/(1-P);
    return (Le + (depth>2 ? (XORShift::RNG()<P ?   // Russian roulette
                                                     LRadiance(reflRay,depth)*RP:f.Multiply(LRadiance(Ray(x,tdir),depth)*TP)) :
                            LRadiance(reflRay,depth)*Re+f.Multiply(LRadiance(Ray(x,tdir),depth)*Tr)))*scaleBy;
}
int main( int argc, char *argv[] ) {

    unsigned int imageWidth     = 500;
    unsigned int imageHeight    = 500;
    unsigned int pixelSamples = argc == 2 ? atoi( argv[1] ) / 4 : 1;

    // Camera
    Ray sceneCamera( Vector3( 50.0, 52.0, 285.0),                   /* Position */
                     Vector3( 00.0,-0.042612 , -1.0).Normalize());  /* Direction */

    Vector3 cx = Vector3(imageWidth * 0.5135 / imageHeight);
    Vector3 cy = (cx % sceneCamera.direction).Normalize() * 0.5135;
    Vector3 r;
    Vector3 *c = new Vector3[imageWidth * imageHeight];

    // OpenMP loop for parallelizing the rendeirng process
    // You have to add the -fopenmp flag to get this support
#pragma omp parallel for schedule(dynamic, 1) private(r)

    // Loop over image rows
    for ( unsigned int y = 0; y < imageHeight; y++ )
    {
        fprintf(stderr,"\n Rendering (%d samples/pixel) %5.2f%%", pixelSamples * 4, 100.0 * y / ( imageHeight - 1 ));
        for ( unsigned short x = 0; x < imageWidth; x++)   // Loop cols

            // Sub-pixel image filteration
            for ( int sy = 0, i = (imageHeight -  y - 1) * imageWidth + x; sy < 2; sy++ )
                for ( int sx = 0; sx < 2; sx++, r = Vector3( ))
                {
                    for ( int s = 0; s < pixelSamples; s++)
                    {
                        double r1 = 2 * XORShift::RNG( );
                        double dx = r1 < 1 ? sqrt( r1 ) - 1: 1 - sqrt( 2 - r1 );
                        double r2 = 2 * XORShift::RNG();
                        double dy = r2 < 1 ? sqrt( r2 ) - 1: 1 - sqrt( 2 - r2 );

                        Vector3 d = cx * (((sx + 0.5 + dx) / 2 + x ) / imageWidth - .5 ) +
                                cy * ((( sy + 0.5 + dy ) / 2 + y ) / imageHeight - .5 ) +
                                sceneCamera.direction;

                        r = r + LRadiance( Ray( sceneCamera.origin + d * 140,
                                                d.Normalize( )), 0 ) * ( 1.0 / pixelSamples );
                    }

                    // Camera rays are pushed ^^^^^ forward to start in interior.
                    c[i] = c[i] + Vector3( Clamp( r.x ), Clamp( r.y ), Clamp( r.z )) * 0.25;
                }
    }

    // Write the image to a ppm file
    FILE *imageFile = fopen( "image.ppm", "w" );

    // Write the header of the image in the following form
    // P3
    // IMAGE_WIDTH IMAGE_HEIGHT
    // IMAGE_DEPTH_PER_PIXEL
    fprintf( imageFile, "P3\n%d %d\n%d\n", imageWidth, imageHeight, 255 );

    // Write image contents to the file
    // Convert the float values of the radiance to int.
    for ( int i = 0; i < imageWidth * imageHeight; i++ )
        fprintf(imageFile,"%d %d %d ",
                ColorToInteger( c[i].x ), ColorToInteger( c[i].y ), ColorToInteger( c[i].z ));
}
