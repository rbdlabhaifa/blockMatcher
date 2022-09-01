extern "C" {
#include <libavutil/motion_vector.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include <iostream>
#include <memory>


int decode_packet(const std::shared_ptr<AVPacket> &pkt, std::shared_ptr<AVCodecContext> &video_dec_ctx,
                  std::shared_ptr<AVFrame> &frame,
                  int *video_frame_count) {
    int ret = avcodec_send_packet(video_dec_ctx.get(), pkt.get());
    if (ret < 0) {
        std::cerr << "Error while sending a packet to the decoder:" << std::endl;
        char error[64];
        av_make_error_string(error, 64, ret);
        std::cerr << error << std::endl;
        return ret;
    }

    while (ret >= 0) {
        ret = avcodec_receive_frame(video_dec_ctx.get(), frame.get());
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            std::cerr << "Error while receiving a frame from the decoder:" << std::endl;
            char error[64];
            av_make_error_string(error, 64, ret);
            std::cerr << error << std::endl;
            return ret;
        }

        if (ret >= 0) {
            int i;
            AVFrameSideData *sd;

            *video_frame_count += 1;
            sd = av_frame_get_side_data(frame.get(), AV_FRAME_DATA_MOTION_VECTORS);
            if (sd) {
                auto *mvs = (const AVMotionVector *) sd->data;
                for (i = 0; i < sd->size / sizeof(*mvs); i++) {
                    const AVMotionVector *mv = &mvs[i];
                    printf("%d,%2d,%2d,%2d,%4d,%4d,%4d,%4d,0x%" PRIx64"\n",
                           *video_frame_count, mv->source,
                           mv->w, mv->h, mv->src_x, mv->src_y,
                           mv->dst_x, mv->dst_y, mv->flags);
                }
            }
            av_frame_unref(frame.get());
        }
    }

    return 0;
}

int open_codec_context(std::shared_ptr<AVFormatContext> &fmt_ctx, enum AVMediaType type,
                       std::shared_ptr<AVCodecContext> &video_dec_ctx,
                       std::string &src_filename, std::shared_ptr<AVStream> &video_stream,
                       std::shared_ptr<int> &video_stream_idx) {
    int ret;
    AVStream *st;
    AVCodecContext *dec_ctx = nullptr;
    const AVCodec *dec = nullptr;
    AVDictionary *opts = nullptr;

    ret = av_find_best_stream(fmt_ctx.get(), type, -1, -1, (AVCodec **) dec, 0);
    if (ret < 0) {
        fprintf(stderr, "Could not find %s stream in input file '%s'\n",
                av_get_media_type_string(type), src_filename.c_str());
        return ret;
    } else {
        int stream_idx = ret;
        st = fmt_ctx->streams[stream_idx];

        dec_ctx = avcodec_alloc_context3(dec);
        if (!dec_ctx) {
            fprintf(stderr, "Failed to allocate codec\n");
            return AVERROR(EINVAL);
        }

        ret = avcodec_parameters_to_context(dec_ctx, st->codecpar);
        if (ret < 0) {
            fprintf(stderr, "Failed to copy codec parameters to codec context\n");
            return ret;
        }

        /* Init the video decoder */
        av_dict_set(&opts, "flags2", "+export_mvs", 0);
        dec = avcodec_find_decoder(AV_CODEC_ID_H264);
        ret = avcodec_open2(dec_ctx, dec, &opts);
        av_dict_free(&opts);
        if (ret < 0) {
            fprintf(stderr, "Failed to open %s codec\n",
                    av_get_media_type_string(type));
            return ret;
        }

        *video_stream_idx = stream_idx;
        video_stream.reset(fmt_ctx->streams[*video_stream_idx]);
        video_dec_ctx.reset(dec_ctx);
    }

    return 0;
}

int clearContext(std::shared_ptr<AVCodecContext> &video_dec_ctx, std::shared_ptr<AVFormatContext> &fmt_ctx,
                 std::shared_ptr<AVFrame> &frame, std::shared_ptr<AVPacket> &pkt, int ret) {
    AVCodecContext *avctx = video_dec_ctx.get();
    avcodec_free_context(&avctx);
    AVFormatContext *s = fmt_ctx.get();
    avformat_close_input(&s);
    AVFrame *f = frame.get();
    av_frame_free(&f);
    AVPacket *p = pkt.get();
    av_packet_free(&p);
    return ret < 0;
}

int main(int argc, char **argv) {
    int ret = 0;
    std::shared_ptr<AVPacket> pkt;
    std::shared_ptr<AVFormatContext> fmt_ctx;
    std::shared_ptr<AVCodecContext> video_dec_ctx;
    std::shared_ptr<AVStream> video_stream;
    std::string src_filename;

    std::shared_ptr<int> video_stream_idx = std::make_shared<int>(-1);
    std::shared_ptr<AVFrame> frame;
    int video_frame_count = 0;


    if (argc != 2) {
        fprintf(stderr, "Usage: %s <video>\n", argv[0]);
        exit(1);
    }
    src_filename = argv[1];
    AVFormatContext *fmt_ctx_tmp;

    if (avformat_open_input(&fmt_ctx_tmp, src_filename.c_str(), nullptr, nullptr) <
        0) {
        fprintf(stderr, "Could not open source file %s\n", src_filename.c_str());
        exit(1);
    }
    fmt_ctx.reset(fmt_ctx_tmp);
    if (avformat_find_stream_info(fmt_ctx.get(), nullptr) < 0) {
        fprintf(stderr, "Could not find stream information\n");
        exit(1);
    }

    open_codec_context(fmt_ctx, AVMEDIA_TYPE_VIDEO, video_dec_ctx, src_filename, video_stream, video_stream_idx);

    av_dump_format(fmt_ctx.get(), 0, src_filename.c_str(), 0);

    if (!video_stream) {
        fprintf(stderr, "Could not find video stream in the input, aborting\n");
        ret = 1;
        return clearContext(video_dec_ctx, fmt_ctx, frame, pkt, ret);
    }

    frame.reset(av_frame_alloc());
    if (!frame) {
        fprintf(stderr, "Could not allocate frame\n");
        ret = AVERROR(ENOMEM);
        return clearContext(video_dec_ctx, fmt_ctx, frame, pkt, ret);
    }

    pkt.reset(av_packet_alloc());
    if (!pkt) {
        fprintf(stderr, "Could not allocate AVPacket\n");
        ret = AVERROR(ENOMEM);
        return clearContext(video_dec_ctx, fmt_ctx, frame, pkt, ret);
    }

    printf("framenum,source,blockw,blockh,srcx,srcy,dstx,dsty,flags\n");

    /* read frames from the file */
    while (av_read_frame(fmt_ctx.get(), pkt.get()) >= 0) {
        if (pkt->stream_index == *video_stream_idx)
            ret = decode_packet(pkt, video_dec_ctx, frame, &video_frame_count);
        av_packet_unref(pkt.get());
        if (ret < 0)
            break;
    }

    /* flush cached frames */
    decode_packet(nullptr, video_dec_ctx, frame, &video_frame_count);


    return ret < 0;

}