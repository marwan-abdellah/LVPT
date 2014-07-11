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
    double Intersect( const Ray &ray,
                      double *tIn = NULL,
                      double *tOut = NULL ) const
    {
        // Fudge factor (epsilon)
        double EPSILON = 1e-4;

        /// A ray with the origin 'o' and direction 'd'
        /// A sphere with know center 'c' and radius 'r'
        /// A point 'p' is on the surface of the sphere

        /// Solve for t where,
        /// ((t * t) * (d . d) +
        /// ((o - c). (2 * t * d)) +
        /// ((o - c).(o - p)) - (R * R) = 0;

        /// The parametric form the of the ray is
        /// ray(t) = o + t * d; t >= 0
        /// We can find t at which the ray intersects the sphere by
        /// the formula A(t * t) + B * t + c = 0, where
        /// A = d.d
        /// B = 2 * (o - c) . d
        /// C = (o - c) (o - c) - (r * r)

        // Vector from the center of the sphere to the origin of the ray.
        // p is the center of the sphere, ( p = c )
        Vector3 oc = position - ray.origin;

        // Compute A, B and C
        // A = 1, becuase the ray is normalized.
        // B = ( 2 * d ) . oc
        // C = ( oc . oc ) - ( radius * radius )
        // discrimanant = B * B - 4 (A * C);
        // discriminant = (( 1 / 4 ) * B * B - C );

        // Let's save some cycles and calculate the reduced factors.
        const double reducedB = oc.DotProduct( ray.direction );

        // Calculate the discriminant
        double sqrtDiscriminant = 0;
        const double discriminant =
                (( reducedB * reducedB ) -
                 oc.DotProduct( oc ) + ( radius * radius ));

        // Check f the discriminant is less than zero or not.
        // If negative, the ray misses the sphere, else intersects with it.
        if( discriminant < 0 )
            return 0;
        else
            sqrtDiscriminant = sqrt( discriminant );

        // If both values of t are negative, then the sphere is
        // behind the ray.
        if ( tIn && tOut )
        {
            if( reducedB - sqrtDiscriminant <= 0 )
                *tIn = 0;
            else
                *tIn = reducedB - sqrtDiscriminant;

            *tOut = ( reducedB + sqrtDiscriminant );
        }

        if( reducedB - sqrtDiscriminant > EPSILON)
            return ( reducedB - sqrtDiscriminant );
        if( reducedB + sqrtDiscriminant > EPSILON)
            return ( reducedB + sqrtDiscriminant );
        return 0;
    }
};


inline double Clamp( const double value )
{
    return value < 0 ? 0 : value > 1 ? 1 : value;
}

inline int ColorToInteger( const double value )
{
    return int( pow( Clamp( value ), 1 / 2.2) * 255 + 0.5 );
}

// Creating a scene that is only composed of spheres.
Sphere sceneSpheres[] =
{
    Sphere( 25.0, Vector3( 27.0, 18.5, 78.0 ), Emission3( 00.0, 00.0, 00.0 ),
    Color3( 1.00, 1.00, 1.00 ) * 0.75, SPEC ),    //Mirr
    Sphere( 12.0, Vector3( 70.0, 43.0, 78.0 ), Emission3( 00.0, 00.0, 00.0 ),
    Color3( 0.87, 0.10, 0.10 ) * 1.00, REFR ),    //Glas
    Sphere( 8.00, Vector3( 55.0, 87.0, 78.0 ), Emission3( 00.0, 00.0, 00.0 ),
    Color3( 1.00, 1.00, 1.00 ) * 0.75, DIFF ),    //Lite
    Sphere( 4.00, Vector3( 55.0, 80.0, 78.0 ), Emission3( 10.0, 10.0, 10.0 ),
    Color3( 00.0, 00.0, 00.0 ) * 1.00, DIFF )     //Lite
};

// Homogenous sphere (participating media)
Sphere homogenousSphere( 300,
                         Vector3( 50.0, 50.0, 80.0 ),
                         Vector3( 00.0, 00.0, 00.0 ),
                         Vector3( 00.0, 00.0, 00.0 ),
                         DIFF );

// Optical properties of the volume
const double sigma_s = 0.009;
const double sigma_a = 0.006;
const double sigma_t = sigma_s + sigma_a;

// Test intersection between ray and spheres in the scene
// and keep the closest intersection.
inline bool Intersects( const Ray &ray,
                        double &tIntsct,
                        int &objId,
                        const double tMax = 1e20 )
{
    // Number of spheres in the scene
    double numSpheres = sizeof( sceneSpheres ) / sizeof( Sphere );
    double itsctDistance;
    double INFINITY_DISTANCE = tIntsct = tMax;

    // Checks if the ray intersects any sphere or not.
    for( int iSphere = int( numSpheres ); iSphere--; )
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

// The _segment_ is the line between the two scattering points
// on the old and the new ray.
inline double SampleSegment( const double epsilon,
                             const float sigma_s,
                             const float segmentMaximumDistance )
{
    return -log( 1.0 - epsilon * ( 1.0 - exp( -sigma_s * segmentMaximumDistance ))) / sigma_s;
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
inline double Scatter( const Ray &ray,      // Input ray
                       Ray *scatteredRay,   // Scattered ray
                       double tIn,          // Where the ray enters the sphere
                       float tOut,          // Where the ray exits the sphere
                       double &segment )    // Segment length
{
    // The segment is the path between two points
    const double segmentMaximumDistance = tOut - tIn;
    segment = SampleSegment(XORShift::RNG(), sigma_s, segmentMaximumDistance);

    Vector3 x = ray.origin + ray.direction * tIn + ray.direction * segment;

    // Sample a direction ~ uniform phase function
    // Vector3 direction = SampleSphere(XORShift::frand( ), XORShift::frand( ));

    // Sample a direction ~ HG phase function
    Vector3 direction = SampleHG(-0.5, XORShift::RNG( ), XORShift::RNG( ));

    // Parameteric variable for determining the intersection point
    Vector3 u,v;

    //
    GenerateOrthoBasis( u, v, ray.direction );

    // New scattering direction
    direction =
            ( u * direction.x ) +
            ( v * direction.y ) +
            ( ray.direction * direction.z );

    // If the ray is scattered, then assign to it the new starting point
    // and the new direction after scatterig.
    if (scatteredRay)
        *scatteredRay = Ray( x, direction );

    // Distance between the two points on the ray segment
    const double distPointsSegment = tOut - tIn;

    // Optical thickness
    const double opticalThickness = sigma_s * distPointsSegment;

    // TODO: Value ?
    const double value = ( 1.0 - exp( -opticalThickness ));

    return value;
}

// Calculate the radiance L at the ray.
Vector3 LRadiance(const Ray &ray, int depth)
{
    // Distance to the intersection
    double distToIntersection;

    // Id of the intersected object
    int intersectedObjID = 0;

    //
    double tNear, tFar;

    //
    double scaleFactor = 1.0;

    //
    double absorptionContribution = 1.0;

    // Find te ray intersection with the homogenous sphere that
    // represents the volume.
    // Checks if the ray intersects the homogenous sphere or not
    const double intersectionDistance =
            homogenousSphere.Intersect(ray, &tNear, &tFar);

    if( intersectionDistance > 0 )
    {
        // This is the scattered ray
        Ray scatteredRay;
        double s;
        double ms = Scatter( ray, &scatteredRay, tNear, tFar, s );
        double prob_s = ms;
        scaleFactor = 1.0 / ( 1.0 - prob_s );


        // Sample a surface or a volume
        if (XORShift::RNG() <= prob_s)
        {
            if ( !Intersects( ray, distToIntersection, intersectedObjID , tNear + s ))
                return LRadiance(scatteredRay, ++depth) * ms * (1.0/prob_s);
            scaleFactor = 1.0;
        }
        else
            if ( !Intersects( ray, distToIntersection, intersectedObjID ))
                return Vector3(0.0, 0.0, 0.0);


        if ( distToIntersection >= tNear )
        {
            const double distInVolume = (distToIntersection > tFar ? tFar - tNear : distToIntersection - tNear);
            absorptionContribution = exp(-sigma_t * distInVolume);
        }
    }
    else
        if ( !Intersects( ray, distToIntersection, intersectedObjID ))
            return Vector3(0.0, 0.0, 0.0);

    // The hit sphere (object)
    const Sphere &obj = sceneSpheres[ intersectedObjID ];

    // Distance until the intersection in the sphere volume3
    // Ray intersection point
    Vector3 intersectionPoint =
            ray.origin + ray.direction * distToIntersection;

    // Get the direction from the intersection point to the
    // center of the sphere.
    // Normal on the surface of the sphere at the
    // intersection point.
    Vector3 normalItsPoint =
            ( intersectionPoint - obj.position ).Normalize();

    // Get the correct direction.
    // Properly orient the surface normal.
    Vector3 normalVector(0.0, 0.0, 0.0);
    if ( normalItsPoint.DotProduct( ray.direction ) < 0 )
        normalVector = normalItsPoint;
    else
        normalVector = normalItsPoint * -1;

    // Get the object color BRDF modulator of the intersected object.
    Vector3 f_BRDF = obj.color;

    // Get the emissive properties of the object.
    Vector3 Le = obj.emission;

    // Maximum reflection ?
    double p = (f_BRDF.x > f_BRDF.y && f_BRDF.x > f_BRDF.z ) ? f_BRDF.x : f_BRDF.y > f_BRDF.z ? f_BRDF.y : f_BRDF.z;


    if( ++depth > 5 )
    {
        if( XORShift::RNG() < p )
            f_BRDF = f_BRDF * ( 1 / p );
        else
            return Vector3( 0.0, 0.0, 0.0 ); // R.R.
    }

    if ( normalItsPoint.DotProduct(normalVector) > 0 || obj.reflection != REFR)
    {
        f_BRDF = f_BRDF * absorptionContribution;
        Le = obj.emission * absorptionContribution; // no absorption inside glass
    }
    else
        scaleFactor = 1.0;


    // Ideal DIFFUSE reflection
    if ( obj.reflection == DIFF )
    {
        // Angle around
        const double r1 = 2 * M_PI * XORShift::RNG( );

        // Distance from center
        const double r2 = XORShift::RNG( );
        double r2s = sqrt(r2);

        // Normal vector(s) 'w', 'u' and 'v' are normal to each others.
        Vector3 w = normalVector;
        Vector3 u = (( fabs( w.x ) > 0.1 ? Vector3( 0,1 ) :
                                           Vector3( 1)) %w ).Normalize( );
        Vector3 v =w % u;

        // Direction of random reflection ray.
        Vector3 reflectionRay = ( u * cos( r1 ) * r2s +
                                  v * sin( r1 ) * r2s +
                                  w * sqrt( 1 - r2 )).Normalize( );

        // Return the radiance emitted from the point Le
        // in addition to the reflected due to the BRDF (scaled with _scaleFactor_)
        const Vector3 LEmission = Le;
        const Vector3 LReflective =
                f_BRDF.Multiply( LRadiance( Ray( intersectionPoint,
                                                 reflectionRay ),
                                            depth ));

        const Vector3 LDiffuse = ( LEmission + LReflective ) * scaleFactor;
        return LDiffuse;
    }
    else
        // Ideal SPECULAR reflection
        if (obj.reflection == SPEC)
        {
            const Vector3 LEmission = Le;
            const Vector3 specularRayDirection =
                    ray.direction - normalItsPoint * 2 *
                    normalItsPoint.DotProduct( ray.direction );
            const Vector3 LReflective =
                    f_BRDF.Multiply(
                        LRadiance( Ray( intersectionPoint,
                                        specularRayDirection), depth ));

            const Vector3 LSpecular = (LEmission + LReflective ) * scaleFactor;
            return LSpecular;
        }

    // Ideal dielectric REFRACTION, glass surface
    Ray reflRay(intersectionPoint,
                ray.direction - normalItsPoint *
                2 * normalItsPoint.DotProduct( ray.direction ));

    // Compute whether the ray is entering or exitting the glass surface.
    bool inOrOut = normalItsPoint.DotProduct(normalVector) > 0;

    // Refractive indices
    // Air refractive index
    double nAmbient = 1;

    // Glass refractive index
    double nGlass = 1.5;

    // Ratio between the refractive indices
    double nChange = inOrOut ?
                ( nAmbient / nGlass ) : ( nGlass / nAmbient );


    double ddn = ray.direction.DotProduct(normalVector);

    // Total internal reflection
    double cos2t;
    if ((cos2t = 1 - nChange * nChange *(1 - ddn * ddn)) < 0)
        return (Le + f_BRDF.Multiply(LRadiance(reflRay,depth)));

    Vector3 tdir = (ray.direction*nChange  - normalItsPoint*((inOrOut?1:-1)*(ddn*nChange +sqrt(cos2t)))).Normalize();
    double a=nGlass-nAmbient;
    double b = nGlass + nAmbient;
    double R0 = a * a / ( b * b );
    double c = 1 - ( inOrOut?-ddn:tdir.DotProduct( normalItsPoint ));
    double Re = R0 +( 1 -R0 ) * c * c * c * c * c;
    double Tr = 1 - Re;
    double P = 0.25 + 0.5 * Re;
    double RP = Re / P;
    double TP = Tr / ( 1 - P );


    // Russian roulette for multiple scattering
    return (Le + (depth>2 ? (XORShift::RNG()<P ? LRadiance(reflRay,depth)*RP:f_BRDF.Multiply(LRadiance(Ray(intersectionPoint,tdir),depth)*TP)) :
                            LRadiance(reflRay,depth)*Re+f_BRDF.Multiply(LRadiance(Ray(intersectionPoint,tdir),depth)*Tr)))*scaleFactor;
}

int main( int argc, char *argv[] )
{
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
        fprintf(stderr,"\n Rendering (%d samples/pixel) %5.2f%%",
                pixelSamples * 4, 100.0 * y / ( imageHeight - 1 ));
        // Loop over image columns
        for ( unsigned short x = 0; x < imageWidth; x++)

            // Sub-pixel image filteration
            for ( int sy = 0, i = (imageHeight -  y - 1) * imageWidth + x; sy < 2; sy++ )
                for ( int sx = 0; sx < 2; sx++, r = Vector3( ))
                {
                    for ( int s = 0; s < pixelSamples; s++)
                    {
                        double r1 = 2 * XORShift::RNG( );
                        double dx = ( r1 < 1 ) ? ( sqrt( r1 ) - 1 ):( 1 - sqrt( 2 - r1 ));
                        double r2 = 2 * XORShift::RNG();
                        double dy = ( r2 < 1 ) ? ( sqrt( r2 ) - 1 ):( 1 - sqrt( 2 - r2 ));

                        Vector3 d = cx * (((sx + 0.5 + dx) / 2 + x ) / imageWidth - .5 ) +
                                cy * ((( sy + 0.5 + dy ) / 2 + y ) / imageHeight - .5 ) +
                                sceneCamera.direction;

                        r = r + LRadiance( Ray( sceneCamera.origin + d * 140,
                                                d.Normalize( )), 0 ) * ( 1.0 / pixelSamples );
                    }

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
