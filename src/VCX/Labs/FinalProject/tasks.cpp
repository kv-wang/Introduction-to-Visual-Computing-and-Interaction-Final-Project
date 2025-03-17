#define GLM_ENABLE_EXPERIMENTAL
#include "Labs/FinalProject/tasks.h"
namespace VCX::Labs::Rendering {

     glm::vec4 GetTexture(Engine::Texture2D<Engine::Formats::RGBA8> const & texture, glm::vec2 const & uvCoord) {
        if (texture.GetSizeX() == 1 || texture.GetSizeY() == 1) return texture.At(0, 0);
        glm::vec2 uv      = glm::fract(uvCoord);
        uv.x              = uv.x * texture.GetSizeX() - .5f;
        uv.y              = uv.y * texture.GetSizeY() - .5f;
        std::size_t xmin  = std::size_t(glm::floor(uv.x) + texture.GetSizeX()) % texture.GetSizeX();
        std::size_t ymin  = std::size_t(glm::floor(uv.y) + texture.GetSizeY()) % texture.GetSizeY();
        std::size_t xmax  = (xmin + 1) % texture.GetSizeX();
        std::size_t ymax  = (ymin + 1) % texture.GetSizeY();
        float       xfrac = glm::fract(uv.x), yfrac = glm::fract(uv.y);
        return glm::mix(glm::mix(texture.At(xmin, ymin), texture.At(xmin, ymax), yfrac), glm::mix(texture.At(xmax, ymin), texture.At(xmax, ymax), yfrac), xfrac);
    }

    glm::vec4 GetAlbedo(Engine::Material const & material, glm::vec2 const & uvCoord) {
        glm::vec4 albedo       = GetTexture(material.Albedo, uvCoord);
        glm::vec3 diffuseColor = albedo;
        return glm::vec4(glm::pow(diffuseColor, glm::vec3(2.2)), albedo.w);
    }

    
    bool cmpx(const Triangle & t1, const Triangle & t2) {
        return t1.center.x < t2.center.x;
    }
    bool cmpy(const Triangle & t1, const Triangle & t2) {
        return t1.center.y < t2.center.y;
    }
    bool cmpz(const Triangle & t1, const Triangle & t2) {
        return t1.center.z < t2.center.z;
    }

    bool IntersectTriangle(Intersection & output, Ray const & ray, glm::vec3 const & p1, glm::vec3 const & p2, glm::vec3 const & p3) {
        // your code here
        glm::vec3 edge1 = p2 - p1, edge2 = p3 - p1;
        glm::vec3 d    = glm::normalize(ray.Direction);
        glm::vec3 pvec = glm::cross(d, edge2);
        float     det  = glm::dot(edge1, pvec);
        if (det < EPS2) return false;
        glm::vec3 tvec = ray.Origin - p1;
        output.u       = glm::dot(tvec, pvec);
        if (output.u < 0.0f || output.u > det) return false;
        glm::vec3 qvec = glm::cross(tvec, edge1);
        output.v       = glm::dot(d, qvec);
        if (output.v < 0.0f || output.u + output.v > det) return false;
        output.t      = glm::dot(edge2, qvec);
        float inv_det = 1.0f / det;
        output.t *= inv_det;
        output.u *= inv_det;
        output.v *= inv_det;
    }
    //上面的沿用lab3的函数

    glm::vec3 SampleLight(float const & roughness, glm::vec3 const & ks, glm::vec3 const & wi, glm::vec3 const & N, float & pdf, bool const & enableImprotanceSampling) {
        static std::random_device                    dev;
        static std::mt19937                          rng(dev());
        static std::uniform_real_distribution<float> rand1(0.0f, 1.0f), rand2(0.0f, 1.0f);
        if (enableImprotanceSampling) {
            if (glm::dot(ks, ks) < EPS1) {
                // cosine-weighted for diffuse
                float     x_1 = rand1(rng), x_2 = rand2(rng);
                float     x = std::cos(2 * PI * x_2) * std::sqrt(x_1);
                float     y = std::sin(2 * PI * x_2) * std::sqrt(x_1);
                float     z = std::sqrt(1 - x_1);
                glm::vec3 localRay(x, y, z);
                pdf = z * std::sqrt(1 - z * z) / PI;
                return toWorld(localRay, N);
            } else {
                // ggx-BRDF for specular
                float     x_1 = rand1(rng), x_2 = rand2(rng);
                float     roughness2 = roughness * roughness;
                float     cos_theta2 = (1 - x_1) / (1 + x_1 * (roughness2 - 1));
                float     sin_theta  = std::sqrt(1 - cos_theta2);
                float     phi        = 2 * PI * x_2;
                float     x          = sin_theta * std::cos(phi);
                float     y          = sin_theta * std::sin(phi);
                float     z          = std::sqrt(cos_theta2);
                float     t          = (roughness2 - 1) * cos_theta2 + 1;
                float     pdf_h      = roughness2 * z * sin_theta / (PI * t * t);
                glm::vec3 localRay(x, y, z);
                glm::vec3 w0 = toWorld(localRay, N);
                glm::vec3 h  = glm::normalize(w0 + wi);
                pdf          = pdf_h / (4 * glm::dot(wi, h));
                return w0;
            }
        } else {
            // uniform sampling
            float     x_1 = rand1(rng), x_2 = rand2(rng);
            float     z = std::fabs(1.0f - 2.0f * x_1);
            float     r = std::sqrt(1.0f - z * z), phi = 2 * PI * x_2;
            glm::vec3 localRay(r * std::cos(phi), r * std::sin(phi), z);
            glm::vec3 w0 = toWorld(localRay, N);
            pdf          = glm::dot(w0, N) > 0 ? 0.5f / PI : 0.0f;
            return w0;
        }
    }

    glm::vec3 toWorld(glm::vec3 const & a, glm::vec3 const & N) {//将局部坐标系中的向量转换到世界坐标系中
        //N is norm in world  coordinate
        // 计算向量 C
        glm::vec3 C;
        if (std::fabs(N.x) > std::fabs(N.y)) {//为了数值稳定
            float invLen = 1.0f / std::sqrt(N.x * N.x + N.z * N.z);
            C = glm::vec3(N.z * invLen, 0.0f, -N.x * invLen);
        } else {
            float invLen = 1.0f / std::sqrt(N.y * N.y + N.z * N.z);
            C = glm::vec3(0.0f, N.z * invLen, -N.y * invLen);
        }

        // 计算向量 B
        glm::vec3 B = glm::cross(C, N);

        // 构造旋转矩阵 R
        glm::mat3 R(B, C, N);

        // 使用逆矩阵进行转换
        glm::vec3 a_world = R * a;

        // 返回转换后的向量
        return a_world;
    }

    bool isLight(glm::vec3 const & pos, VCX::Engine::Light const & light) {//判断 pos 是否在光源 light 的范围内
        if (std::abs(pos.y - light.Position.y) < 1 && std::abs(pos.x - light.Position.x) < 65 && std::abs(pos.z - light.Position.z) < 57.5)
            return true;
        return false;
    }

    float schlick(float const & ratio, glm::vec3 const & w, glm::vec3 const & N) {
        //通过schlick近似方法计算Fresnel Term
        float R_theta = (1-ratio)/(1+ratio);
        R_theta = R_theta * R_theta;
        float cos_theta = glm::dot(w, N);
        return R_theta + (1-R_theta)*std::pow((1-cos_theta),5);
    }

    float D_ggx(float const & roughness, glm::vec3 const & h, glm::vec3 const & N) {
        float cos_theta   = glm::dot(N, h);
        cos_theta = std::max(cos_theta, 0.0f);
        float cos_theta_2 = cos_theta * cos_theta;
        float roughness2  = roughness * roughness;
        float t = cos_theta_2 * (roughness2 - 1) + 1;
        float d = roughness2 / (PI * t * t);
        return d;
    }
    float D_Beckmann(float const & roughness, glm::vec3 const & h, glm::vec3 const & N) {
        float cos_theta = glm::dot(N, h); // 计算半向量 h 与法向量 N 的点积，并取最大值为 0
        cos_theta = std::max(cos_theta, 0.0f);
        float cos_theta_2 = cos_theta * cos_theta; // 计算 cos_theta 的平方
        float roughness2 = roughness * roughness; // 计算粗糙度的平方
        float exponential = (cos_theta_2 - 1) / (roughness2 * cos_theta_2); // 计算指数项
        float d = std::exp(exponential) / (PI * roughness2 * cos_theta_2 * cos_theta_2); // 计算 Beckmann 分布函数
    return d;
    }
   
    
    float Disney(float const & roughness, glm::vec3 const & w, glm::vec3 const & half){
        //Disney几何函数
        float k = (0.5f + 0.5f*roughness);
        k = k * k;
        float dot_product = glm::dot(w, half);
        return 2*dot_product/(dot_product + sqrtf(k*k + (1.0f-k*k)*dot_product*dot_product));
    }

    glm::vec3 Sample(VCX::Engine::Light const & lightSource) {
       
        static std::random_device randomDevice;
        static std::mt19937 randomGenerator(randomDevice());
        static std::uniform_real_distribution<float> uniformDistribution(0.0f, 1.0f);

        float randX = uniformDistribution(randomGenerator);
        float randZ = uniformDistribution(randomGenerator);

        // 计算光源区域内的随机点
        glm::vec3 randomPoint = lightSource.Position;
        randomPoint.x += (130.0f * randX - 65.0f);
        randomPoint.z += (115.0f * randZ - 57.5f);

        return randomPoint;
    }

    glm::vec3 BRDF(float const & roughness, float const & shininess, glm::vec3 const & kd, glm::vec3 const & ks, glm::vec3 const & wi, glm::vec3 const & w0, glm::vec3 const & N) {
        /*
        roughness：表面的粗糙度，影响微表面的分布。
        shininess：表面的高光系数，影响镜面反射的锐利度。
        kd：漫反射颜色。
        ks：镜面反射颜色。
        wi：入射光线方向，是一个归一化的向量。
        w0：反射光线方向，是一个归一化的向量。
        N：表面的法向量，是一个归一化的向量。*/
        glm::vec3 diffuse = glm::dot(N, w0) > 0.0f ? kd / PI : glm::vec3(0.0f);
        if (glm::dot(ks, ks) < 1e-2f) 
            return diffuse; //没有镜面反射成分
        glm::vec3 h = glm::normalize(wi + w0);
        float f = schlick(shininess, wi, N);
        float g = Disney(roughness, wi, h) * Disney(roughness, w0, h);
        float d = D_ggx(roughness, h, N);
        
        glm::vec3 specular = ks*g*d / std::max((4 * glm::dot(wi, N) * glm::dot(w0, N)), 0.0001f);
        return (1.0f - f) * diffuse + f * specular;
    }


    glm::vec3 PathTrace(const RayIntersector & intersector, Ray ray, bool const & enableImprotanceSampling) {
        const float P_RR = 0.75f; // 俄罗斯轮盘赌的阈值，减小递归次数
        glm::vec3 accumulatedColor(0.0f);
        RayHit intersection = intersector.IntersectRay(ray);
        if (!intersection.IntersectState) return accumulatedColor;

        glm::vec3 rayDirection = glm::normalize(-ray.Direction);
        glm::vec3 hitPosition = intersection.IntersectPosition;
        glm::vec3 surfaceNormal = intersection.IntersectNormal;
        glm::vec3 diffuseColor = intersection.IntersectAlbedo;
        glm::vec3 specularColor = intersection.IntersectMetaSpec;
        float transparency = intersection.IntersectAlbedo.w;
        float shininess = intersection.IntersectMetaSpec.w * 256;
        VCX::Engine::Light lightSource = intersector.InternalScene->Lights[0];
        glm::vec3 lightDirection(0, -1, 0); // 灯光向下打

        if (isLight(hitPosition, lightSource)) {
            return lightSource.Intensity;
        }

        const float surfaceRoughness = 0.15f;
        
        const float lightPdf = 1.0f / (130 * 115);
        glm::vec3 sampledLightPosition = Sample(lightSource);
        glm::vec3 lightVector = sampledLightPosition - hitPosition;
        glm::vec3 normalizedLightDir = glm::normalize(lightVector);
        float lightAttenuation = 1.0f / glm::dot(lightVector, lightVector);

        Ray shadowRay(sampledLightPosition, -normalizedLightDir);
        RayHit shadowIntersection = intersector.IntersectRay(shadowRay);
        float shadowDistance = glm::distance(hitPosition, shadowIntersection.IntersectPosition);

        while (shadowIntersection.IntersectState && shadowDistance > 1e-2f) {
            if (shadowIntersection.IntersectAlbedo.w >= 0.2f) break;
            shadowIntersection = intersector.IntersectRay(Ray(shadowIntersection.IntersectPosition, -normalizedLightDir));
            shadowDistance = glm::distance(hitPosition, shadowIntersection.IntersectPosition);
        }

        if (!shadowIntersection.IntersectState || shadowDistance <= 1e-2f) {
            glm::vec3 brdfValue = BRDF(surfaceRoughness, shininess, diffuseColor, specularColor, rayDirection, normalizedLightDir, surfaceNormal);
            accumulatedColor += lightSource.Intensity * brdfValue * glm::dot(normalizedLightDir, surfaceNormal) * std::max(0.0f, glm::dot(-normalizedLightDir, lightDirection)) / lightPdf * lightAttenuation;
        }

        static std::random_device randomDevice;
        static std::mt19937 randomGenerator(randomDevice());
        static std::uniform_real_distribution<float> randomDistribution(0.0f, 1.0f);

        if (randomDistribution(randomGenerator) > P_RR) {
            return accumulatedColor;
        }

        float pdf = 0.0f;
        glm::vec3 nextDirection = glm::normalize(SampleLight(surfaceRoughness, specularColor, rayDirection, surfaceNormal, pdf, enableImprotanceSampling));
        Ray nextRay(hitPosition, nextDirection);
        RayHit nextIntersection = intersector.IntersectRay(nextRay);
        glm::vec3 nextPosition = nextIntersection.IntersectPosition;

        if (nextIntersection.IntersectState && !isLight(nextPosition, lightSource)) {
            glm::vec3 brdfValue = BRDF(surfaceRoughness, shininess, diffuseColor, specularColor, rayDirection, nextDirection, surfaceNormal);
            accumulatedColor += PathTrace(intersector, nextRay, enableImprotanceSampling) * brdfValue * glm::dot(nextDirection, surfaceNormal) / pdf / P_RR;
        }

        return accumulatedColor;
    }
} // namespace VCX::Labs::Rendering