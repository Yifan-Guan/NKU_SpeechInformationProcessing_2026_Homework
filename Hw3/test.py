import torch
import librosa
import soundfile as sf
from ns3_codec import FACodecEncoder, FACodecDecoder

def main():
    timbre_reconstruct = True

    fa_encoder = FACodecEncoder(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )

    fa_decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )

    fa_encoder.load_state_dict(torch.load("./ckpt/ns3_facodec_encoder.bin"))
    fa_decoder.load_state_dict(torch.load("./ckpt/ns3_facodec_decoder.bin"))

    
    def load_audio(wav_path):
        wav = librosa.load(wav_path, sr=16000)[0]
        wav = torch.from_numpy(wav).float()
        wav = wav.unsqueeze(0).unsqueeze(0)
        return wav

    
    test_wav_file = "speaker1"
    timbre_wav_file = "speaker"
    test_wav = load_audio(f"samples/{test_wav_file}.wav")
    timbre_wav = load_audio(f"samples/{timbre_wav_file}.wav")
    print("Test Audio Shape: ", test_wav.shape)

    with torch.no_grad():

        encoder_out = fa_encoder(test_wav)
        timbre_encoder_out = fa_encoder(timbre_wav)
        print("Encoder_out Shape: ", encoder_out.shape)

        _, vq_id, _, _, spk_embs = fa_decoder(encoder_out, eval_vq=False, vq=True)
        prosody_code = vq_id[:1]
        print("Prosody Code Shape:", prosody_code.shape)
        cotent_code = vq_id[1:3]
        print("Cotent Code Shape:", cotent_code.shape)
        detail_code = vq_id[3:]
        print("Residual Code Shape:", detail_code.shape) 
        
        _, _, _, _, spk_embs = fa_decoder(encoder_out, eval_vq=False, vq=True)
        _, _, _, _, timbre_spk_embs = fa_decoder(timbre_encoder_out, eval_vq=False, vq=True)
        _, _, _, quantized = fa_decoder.quantize(encoder_out)
        prosody = quantized[0]
        print("Prosody Embedding Shape:", prosody.shape)
        content = quantized[1]
        print("Content Embedding Shape:", content.shape)
        detail = quantized[2]
        print("Detail Embedding Shape:", detail.shape)
        spk_embs = spk_embs
        print("Speaker Embedding Shape:", spk_embs.shape)
        
        all_embs = prosody + content + detail
        if timbre_reconstruct:
            rec_wav = fa_decoder.inference(all_embs, timbre_spk_embs)
        else:
            rec_wav = fa_decoder.inference(all_embs, spk_embs)

    print("Reconstruct Audio Shape: ", rec_wav.shape)
    if timbre_reconstruct:
        sf.write(f"samples/{test_wav_file}_timbre_rec.wav", rec_wav[0][0].cpu().numpy(), 16000)
    else:
        sf.write(f"samples/{test_wav_file}_rec.wav", rec_wav[0][0].cpu().numpy(), 16000)
    print("Successfully Reconstruct!")

if __name__ == "__main__":
    main()