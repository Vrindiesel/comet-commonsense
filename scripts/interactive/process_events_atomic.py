import os
import sys
import argparse
import torch
import json

from progress.bar import Bar
sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive


def run_dir_files(category, data_loader, indir, model, outdir, sampler, text_encoder):
    fnames = list(os.listdir(indir))
    num_files = len(fnames)
    for j, fname in enumerate(fnames):
        print("processing file {}/{}:".format(j + 1, num_files), fname)
        with open(os.path.join(indir, fname), "r") as fin:
            d = json.load(fin)

        bar = Bar("Processing ", max=len(d["sentences"]))
        for sent in d["sentences"]:
            for event in sent:
                input_event = event["string"]
                # print(" * annotating event:", input_event)

                outputs = interactive.get_atomic_sequence(
                    input_event, model, sampler, data_loader, text_encoder, category, verbos=False)
                event["annotations"] = outputs

                # input(">>>")
            bar.next()
        bar.finish()

        outpath = os.path.join(outdir, fname)
        with open(outpath, "w") as fout:
            json.dump(d, fout, indent=2)

        break

"""
Effects on others:
As a result, others feel: oReact
As a result, others want: oWant
Others then: oEffect

Causes for PersonX
Because PersonX wanted: xIntent
Before, PersonX needed: xNeed

Effects on PersonX
As a result, PersonX feels: xReact
As a result, PersonX wants: xWant
PersonX then: xEffect

Attributes of PersonX
PersonX is seen as: xAttr


"""

def main():

    indir = "../Plan-and-write/common/sample_outputs/stories/gold_storylines_test_1000"

    infile = "../Plan-and-write/common/model_outputs/unpacked/WP_title2storyline_glove_nomask_19994-event/all-stories.json"

    model_file = "pretrained_models/atomic_pretrained_model.pickle"
    outdir = "outputs"


    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="models/atomic-generation/iteration-500-50000/transformer/categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40542/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_1000-es_1000-categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/6.25e-05_adam_64_22000.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="help")
    args = parser.parse_args()


    args.model_file = model_file
    args.device = "0"

    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("atomic", opt)

    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"



    input_event = None

    # one of topk-n, beam-n, greedy
    sampling_algorithm = "topk-5"
    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)

    category = "all"
    category = ["xEffect", "oEffect"]
    #run_dir_files(category, data_loader, indir, model, outdir, sampler, text_encoder)

    with open(infile, "r") as fin:
        story_list = json.load(fin)


    done_storyies = []
    for j, story in enumerate(story_list):
        print("processing story {}/{}:".format(j + 1, len(story_list)))

        for event_source in ["pred_events", "gold_events"]:

            bar = Bar("Processing {}".format(event_source), max=len(story[event_source]))

            for event in story[event_source]:
                input_event = event["string"]
                #print(" * annotating event:", input_event)

                outputs = interactive.get_atomic_sequence(
                    input_event, model, sampler, data_loader, text_encoder, category, verbos=False)
                event["annotations"] = outputs

                # input(">>>")
                bar.next()
            bar.finish()

        done_storyies.append(story)
        break


    outpath = os.path.join(outdir, "all-stories.json")
    with open(outpath, "w") as fout:
        json.dump(done_storyies, fout, indent=2)






if __name__ == "__main__":
    from time import time

    #main()
    t0 = time()
    main()
    cum = (time() - t0) / 60
    print("elapsed minutes: {:.2f}".format(cum))

