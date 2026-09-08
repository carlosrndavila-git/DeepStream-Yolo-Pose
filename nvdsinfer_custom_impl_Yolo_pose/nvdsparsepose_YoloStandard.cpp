// Standard YOLO11 pose: FLOAT [56,N], xywh, score, 17*(x,y,visibility).
// Supported profile: confidence=.25, IoU=.5, visibility=.3, cap=300.
// Equal scores preserve candidate index. Payload v1 is 60 floats, width=60,height=1.
#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
constexpr unsigned kPayload = 60, kMax = 300;
struct Candidate { unsigned index; float cx, cy, w, h, score; };
double iou(const Candidate& a, const Candidate& b) {
    const double left = std::max(double(a.cx)-a.w/2.0, double(b.cx)-b.w/2.0);
    const double top = std::max(double(a.cy)-a.h/2.0, double(b.cy)-b.h/2.0);
    const double right = std::min(double(a.cx)+a.w/2.0, double(b.cx)+b.w/2.0);
    const double bottom = std::min(double(a.cy)+a.h/2.0, double(b.cy)+b.h/2.0);
    const double area = std::max(0.0,right-left)*std::max(0.0,bottom-top);
    const double total = double(a.w)*a.h + double(b.w)*b.h - area;
    return total > 0 ? area/total : 0;
}
}
extern "C" bool NvDsInferParseYolo11PoseStandard(
    const std::vector<NvDsInferLayerInfo>& layers, const NvDsInferNetworkInfo& net,
    const NvDsInferParseDetectionParams& params, std::vector<NvDsInferInstanceMaskInfo>& output) {
    try {
        if(net.width != 640 || net.height != 640 || params.numClassesConfigured != 1 ||
           params.perClassPreclusterThreshold.size() != 1 || params.perClassPreclusterThreshold[0] != .25f)
            throw std::runtime_error("unsupported pose operating point");
        const NvDsInferLayerInfo* selected = nullptr;
        for(const auto& layer : layers) {
            if(layer.inferDims.numDims == 2 && layer.inferDims.d[0] == 56) {
                if(selected) throw std::runtime_error("ambiguous pose outputs");
                selected=&layer;
            }
        }
        if(!selected || selected->dataType != FLOAT || !selected->buffer ||
           selected->inferDims.d[1] < 1 || selected->inferDims.d[1] > 8400)
            throw std::runtime_error("expected FLOAT [56,N] pose output");
        const unsigned count=selected->inferDims.d[1];
        const float* data=static_cast<const float*>(selected->buffer);
        if(selected->inferDims.numElements != 56*count) throw std::runtime_error("pose element count mismatch");
        std::vector<Candidate> candidates;
        for(unsigned n=0; n<count; ++n) {
            for(unsigned c=0; c<56; ++c) if(!std::isfinite(data[c*count+n]))
                throw std::runtime_error("nonfinite pose output");
            const float score=data[4*count+n];
            if(score < .25f) continue;
            Candidate c{n,data[n],data[count+n],data[2*count+n],data[3*count+n],score};
            if(score > 1 || c.w <= 0 || c.h <= 0) throw std::runtime_error("invalid pose candidate");
            candidates.push_back(c);
        }
        std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a,const Candidate& b){return a.score>b.score;});
        std::vector<Candidate> keep;
        for(const auto& c:candidates) {
            bool suppressed=false;
            for(const auto& old:keep) if(iou(c,old)>.5) {suppressed=true;break;}
            if(!suppressed) keep.push_back(c);
            if(keep.size()==kMax) break;
        }
        std::vector<std::unique_ptr<float[]>> owned;
        std::vector<NvDsInferInstanceMaskInfo> result;
        for(const auto& c:keep) {
            std::unique_ptr<float[]> payload(new float[kPayload]{});
            payload[0]=1;payload[1]=net.width;payload[2]=net.height;payload[3]=c.index;
            payload[4]=c.cx;payload[5]=c.cy;payload[6]=c.w;payload[7]=c.h;payload[8]=c.score;
            for(unsigned k=0;k<51;++k) {
                float value=data[(5+k)*count+c.index];
                if(k%3==2) {
                    if(value<0 || value>1) throw std::runtime_error("invalid visibility");
                    if(value<.3f) value=0;
                }
                payload[9+k]=value;
            }
            NvDsInferInstanceMaskInfo obj{};
            obj.classId=0;obj.detectionConfidence=c.score;
            obj.left=c.cx-c.w/2;obj.top=c.cy-c.h/2;obj.width=c.w;obj.height=c.h;
            obj.mask=payload.get();obj.mask_width=kPayload;obj.mask_height=1;obj.mask_size=kPayload*sizeof(float);
            result.push_back(obj);owned.push_back(std::move(payload));
        }
        output.reserve(output.size()+result.size());
        output.insert(output.end(),result.begin(),result.end());
        for(auto& payload:owned) payload.release(); // NvDsInferContext deletes accepted allocations.
        return true;
    } catch(const std::exception& e) {
        std::cerr << "YOLO11 pose parser: " << e.what() << std::endl;
        return false;
    }
}
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo11PoseStandard);
