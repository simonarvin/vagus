
import numpy as np
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from constants import *
from arguments import Arguments
from scipy import stats
import ppscore as pps
from datetime import datetime
from heartrate3 import HR
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import matplotlib.patches as patches
import kaplanmeier as km
from sample_vid import Sample_vid

from dominance_analysis import Dominance

plt.rc('axes', labelsize=10, titlesize=10)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

fontdict={'fontsize': 12,
'fontweight' : "bold",
'verticalalignment': 'baseline',
'horizontalalignment': "left"}

plt.rcParams.update({
  "font.family": "Arial"
})


TYPE = "analysis"

args = Arguments(type = TYPE)
args = args.args()
path = args.path
HR_ON = args.no_hr
HR_skip = args.retain_hr


included_IDS = []
dists_=[]
time_=[]
stims_ = []
treatments_ = []
weights_ = []
genders_=[]
ages_ = []
HRs_ = []
marcain_deaths=[]
control_deaths = []
times_ = []

data_dirs = [ data_dir for data_dir in os.listdir(path) if os.path.isdir(os.path.join(path, data_dir)) and not ("-" in data_dir) ]

for dir in data_dirs:
    dir = f"{path}/{dir}"
    overview = pd.read_excel(f"{dir}/trial_overview.xlsx", sheet_name="Sheet1", engine='openpyxl')
    inclusions = overview["Include"].values
    IDs = overview["Experiment ID"].values
    treatments = overview["Treatment"].values
    inclusions = overview["Include"].values
    genders = overview["Gender"].values
    weights = overview["Weight (g)"].values
    deaths = np.array(overview["Note"].values, dtype=str)
    #print(deaths)

    days_from_start = (pd.to_datetime(str(IDs[1])[:8], format = "%Y%m%d") - pd.to_datetime(start_date, format = "%Y%m%d")).days

    deaths = np.core.defchararray.find(np.char.lower(deaths), "died") != -1 #or np.flatnonzero(np.core.defchararray.find(deaths, "died")!=-1)
    #print(deaths, treatments)
    #print(treatments[deaths])
    marcain_deaths.append(np.array([True for t in treatments[deaths] if t == "Marcain"]).sum())
    control_deaths.append(np.array([True for t in treatments[deaths] if t == "Control"]).sum())
    times_.append(days_from_start)

    birthdates = pd.to_datetime(overview["Birthdate"].values, unit='ms').to_pydatetime()


    for i, ID in enumerate(IDs):
        if inclusions[i] == 0 or pd.isna(ID):
            continue
        #if ID != "20211201-181322":
        #    continue

        treatments_.append(treatments[i])
        weights_.append(weights[i])
        genders_.append(genders[i])



        age = round((datetime.strptime(ID, '%Y%m%d-%H%M%S') - birthdates[i]).days/7)

        ages_.append(age)
        included_IDS.append(ID)

        trial_dir = f"{dir}/{ID}"
        trial_data = f"{dir}/{ID}/vids"

        with open(f"{trial_dir}/datalog.log", "r") as log_file:
            log_lines = np.array(log_file.readlines(), dtype=np.float64)

        with open(f"{trial_dir}/stimlog.log", "r") as stim_file:
            stim_lines = stim_file.readlines()

        HR_processor = HR(trial_dir, ID = ID, ON = HR_ON, skip = HR_skip)
        sample_vid = Sample_vid(trial_data, SAMPLE)
        time_.append(log_lines)
        stim_indices = []

        stim_lines.append(float(stim_lines[0]) - offset)
        stim_lines = np.array(stim_lines, dtype=np.float64)
        stims_.append(stim_lines)

        for stim in stim_lines:
            stim_indices.append(find_nearest(log_lines, stim))

        print(f"datalogs read, id = {ID}")
        dists = []
        for cam in cameras:
            df = pd.read_hdf(f"{trial_data}/{ID}_cam_{cam}{dlc_suffix}") #load dataframe

            #load body-markers:
            snout = df.xs('snout', level = 'bodyparts', axis = 1).to_numpy()
            leftear = df.xs('leftear', level = 'bodyparts', axis = 1).to_numpy()
            rightear = df.xs('rightear', level = 'bodyparts', axis = 1).to_numpy()
            upperspine = df.xs('upperspine', level = 'bodyparts', axis = 1).to_numpy()

            midspine = df.xs('midspine', level = 'bodyparts', axis = 1).to_numpy()
            disttail = df.xs('disttail', level = 'bodyparts', axis = 1).to_numpy()
            lowerspine= df.xs('lowerspine', level = 'bodyparts', axis = 1).to_numpy()

            #compute multiplicative confidence score
            score =  upperspine[:, 2] * leftear[:, 2] * rightear[:, 2] #*spine[:,2]* midspine[:, 2]

            midspine_score = midspine[:, 2]
            #reformat body-marker, keep x,y dimensions
            snout, upperspine, leftear, rightear, midspine, lowerspine, disttail = snout[:, :2], upperspine[:, :2], leftear[:, :2], rightear[:, :2], midspine[:, :2], lowerspine[:, :2], disttail[:, :2]
            sample_vid.add_tuple((snout, upperspine, leftear, rightear, midspine, lowerspine, disttail), f"{trial_data}/{ID}_cam_{cam}.mp4")
            #filter and interpolate positions according to score cutoff
            #print(f"filtering coordinates, cutoff = {cutoff}")
            excl, x = cutoff_condition(score, cutoff)

            for element in [upperspine, leftear, rightear]:
                element[excl, 1] = np.interp(x(excl), x(~excl), element[:, 1][~excl])
                element[excl, 0] = np.interp(x(excl), x(~excl), element[:, 0][~excl])


            HR_processor.load(cam, midspine, midspine_score, log_lines)

            euc_mean =  (upperspine + leftear + rightear)/3 #calculate euclidean mean = center of head

            #calculate the speed of movement of the head:
            euc_mean_change = np.diff(euc_mean, axis = 0)
            time_diff = np.diff(log_lines,axis=0)

            dist = np.linalg.norm(euc_mean_change, axis=1)/time_diff #speed

            #add the speed value for each frame, for each camera
            if len(dists) == 0:
                dists = dist
            else:
                dists += dist

        dists_.append(dists/len(cameras)) #mean speed per frame (across cameras)

        HRs_.append(HR_processor.compute())

        #if i == 1:
        #    break
    #break

sample_vid.generate()

fig = plt.figure(figsize=(12, 8))
outer = gridspec.GridSpec(5, 4, wspace = 0.2, hspace = 0.2)
fig.canvas.set_window_title('All traces')

fig_ = plt.figure(figsize=(6, 7))
outer_ = gridspec.GridSpec(2, 1, wspace = 0.2, hspace = 0.3)
fig_.canvas.set_window_title('Sample traces')
#fig, axs = plt.subplots(4, 4, figsize =(6, 4))
#axs_ = axs.reshape(-1)
#plt.tight_layout()

kaplan_df = pd.DataFrame()
kaplan_df["time"] = np.array(times_).astype(int)
kaplan_df["marcain"] = np.array(marcain_deaths).astype(int)
kaplan_df["control"] = np.array(control_deaths).astype(int)

print(kaplan_df, kaplan_df["marcain"].sum())


onsets = []
colors_ = []
marcain = []
control = []
marcain_HRs = []
control_HRs = []
marcain_HRs_std = []
control_HRs_std = []

for i, dist in enumerate(dists_):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=outer[i], wspace = 0.1, hspace = 0.2, height_ratios = [2.5, 1])
    ax, ax_HR = plt.Subplot(fig, inner[0]), plt.Subplot(fig, inner[1])

    smooth_dist = moving_average(dist, 50) #smooth out noise using the moving average

    threshold = np.mean(smooth_dist) * .8 + np.amin(smooth_dist) * 2 #compute the threshold for "no-motion"

    dist_bool = (smooth_dist < threshold).astype(int) #check which entries are below threshold = not moving.
    #find clusters of no-motion:
    dist_labeled = measure.label(dist_bool, connectivity=1)
    idx = [np.where(dist_labeled == label)[0]
       for label in np.unique(dist_labeled)
       if label]

    anesthetized = max(idx, key=len) #assumption: largest no-motion cluster = anesthetic period

    onsets.append(time_[i][anesthetized[0]] - stims_[i][0])

    treatment = treatments_[i]
    #print(time_[i][anesthetized[0]], stims_[i][0], included_IDS[i], treatment)

    if treatment != "Control":
        color = colours[0]
        marcain.append(onsets[-1])
    else:
        control.append(onsets[-1])
        color= colours[1]

    colors_.append(color)
    time_c = time_[i] - time_[i][0]

    ax.axvline(time_c[anesthetized[0]], linestyle="--", lw=1, alpha=.3,color="k")
    ax.axvline(time_c[anesthetized[-1]], linestyle="--", lw = 1,alpha=.3,color="k")
    ax.plot(time_c[:-(len(time_[i])-len(smooth_dist))], smooth_dist, lw = 1, color=color)

    x_ON = stims_[i][0] - time_[i][0]
    x_OFF = stims_[i][1] - time_[i][0]

    if HR_ON:
        HR_llb, HR_ulb= 200, 340
        HR_ll, HR_ul = 100, 420

        x_HR = np.linspace(time_c[0], time_c[-(len(time_[i])-len(smooth_dist))], len(HRs_[i]))

        pre_HR_idx = np.argmax(x_HR > x_ON)
        post_HR_idx = np.argmax(x_HR > x_OFF)

        pre_HR = HRs_[i][:pre_HR_idx]
        post_HR = HRs_[i][post_HR_idx:]
        per_HR = HRs_[i][pre_HR_idx:post_HR_idx]

    #    marcain_HRs_std.append([np.std(pre_HR), np.std(per_HR), np.std(post_HR)]) if treatment == "Marcain" else control_HRs_std.append([np.std(pre_HR), np.std(per_HR), np.std(post_HR)])

        marcain_HRs.append([np.mean(pre_HR), np.mean(per_HR), np.mean(post_HR)]) if treatment == "Marcain" else control_HRs.append([np.mean(pre_HR), np.mean(per_HR), np.mean(post_HR)])
        HRs_[i] = moving_average(HRs_[i], 50)

        ax_HR.plot(x_HR, HRs_[i], lw=1, color="grey")
        subfig=["A", "B"]
        for_fig = [0,5]
        if i in for_fig:
            idx = for_fig.index(i)
            inner_ = gridspec.GridSpecFromSubplotSpec(2, 1,
                            subplot_spec=outer_[idx], wspace = 0.1, hspace = 0.2, height_ratios = [2.5, 1])
            ax_, ax_HR_ = plt.Subplot(fig_, inner_[0]), plt.Subplot(fig_, inner_[1])

            ax_HR_.plot(x_HR, HRs_[i], lw=1, color="grey")

            ax_.axvline(x_ON, linestyle="--",lw=1, color="k", zorder=99997) #isoflurane ON
            ax_.axvline(x_OFF, linestyle="--",lw=1, color="k", zorder=99996) #isoflurane OFF

            ax_HR_.axvline(x_ON, linestyle="--",lw=1, color="k", zorder=99997) #isoflurane ON
            ax_HR_.axvline(x_OFF, linestyle="--",lw=1, color="k", zorder=99996) #isoflurane OFF

            #ax_.set_xticks([stims_[i][0] - time_[i][0],stims_[i][1] - time_[i][0],time_c[anesthetized[0]],time_c[anesthetized[-1]]], minor=True)
            ax_.set_xticklabels([])

            ax_.set_xlim(stims_[i][0] - time_[i][0] - offset, time_c[-1])
            ax_.set_ylim(0, 280)
            ax_.set_yticks([0,200])


            #ax_HR_.set_xlim(stims_[i][0] - time_[i][0] - offset, time_c[-1])


            ax_HR_.set_ylim(HR_ll, HR_ul)

            ax_.set_xlim(stims_[i][0] - time_[i][0] - offset, stims_[i][0] - time_[i][0] - offset + 150)
            ax_HR_.set_xlim(stims_[i][0] - time_[i][0] - offset, stims_[i][0] - time_[i][0] - offset + 150)
            ax_.plot(time_c[:-(len(time_[i]) - len(smooth_dist))], smooth_dist, lw = 1, color = color)
            ax_.axvline(time_c[anesthetized[0]], linestyle = "--", lw = 1, alpha = .3, color = "k", zorder=99999)
            ax_.axvline(time_c[anesthetized[-1]], linestyle = "--", lw = 1,alpha = .3, color = "k", zorder=99998)

            ax_HR_.axvline(time_c[anesthetized[0]], linestyle = "--", lw = 1, alpha = .3, color = "k", zorder=99999)
            ax_HR_.axvline(time_c[anesthetized[-1]], linestyle = "--", lw = 1,alpha = .3, color = "k", zorder=99998)


            ax_.set_title(subfig[idx], fontdict=fontdict, loc="left",x=-.1)
            plt.setp(ax_.get_xticklabels(), visible = False)
            fig_.add_subplot(ax_)
            fig_.add_subplot(ax_HR_)

            ax_inset = fig_.add_axes([0, 0, 1, 1])# plt.axes([0, 0, 1, 1])
            h_inset = .35
            spacing = .05
            ip = InsetPosition(ax_, [1 - h_inset - spacing/2, (1 - h_inset - spacing * 4), h_inset, h_inset])

            #_, (ax_inset, ax_inset2) = plt.subplots(2)
            ax_inset.set_axes_locator(ip)

            ax_inset.plot(time_c[:-(len(time_[i])-len(smooth_dist))], smooth_dist, lw = 1, color=color)
            ax_inset.set_xlim(stims_[i][0] - time_[i][0] - offset, time_c[-1])
            ax_inset.set_ylim(0, 250)

            ax_inset.axvline(x_ON, linestyle = "--", lw=1, color="k") #isoflurane ON
            ax_inset.axvline(x_OFF, linestyle = "--", lw=1, color="k") #isoflurane OFF

            ax_inset.axvline(time_c[anesthetized[0]], linestyle="--", lw=1, alpha=.3,color="k")
            ax_inset.axvline(time_c[anesthetized[-1]], linestyle="--", lw = 1,alpha=.3,color="k")


            ax_inset.set_xticklabels([])
            ax_inset.set_yticklabels([])


            ax_inset_l = fig_.add_axes([0, 0, 1, 1])# plt.axes([0, 0, 1, 1])

            ip_l = InsetPosition(ax_, [1 - h_inset - spacing/2, 1 - h_inset - spacing * 4 - h_inset/3 - spacing, h_inset, h_inset/3])
            ax_inset_l.set_axes_locator(ip_l)
            ax_inset_l.plot(x_HR, HRs_[i], lw=1, color="grey")

            #ax_inset_l.set_xticklabels([])
            ax_inset_l.set_yticklabels([])
            ax_inset_l.set_ylim(HR_ll, HR_ul)
            ax_inset_l.set_xlim(stims_[i][0] - time_[i][0] - offset, time_c[-1])
            ax_inset_l.tick_params('both', length=2, width=1, which='major')
            ax_inset.tick_params('both', length=2, width=1, which='major')

            ax_inset_l.axvline(x_ON, linestyle = "--", lw=1, color="k") #isoflurane ON
            ax_inset_l.axvline(x_OFF, linestyle = "--", lw=1, color="k") #isoflurane OFF

            ax_inset_l.axvline(time_c[anesthetized[0]], linestyle="--", lw=1, alpha=.3,color="k")
            ax_inset_l.axvline(time_c[anesthetized[-1]], linestyle="--", lw = 1,alpha=.3,color="k")

            top_xax = ax_inset.twiny()
            top_xax.set_xticks([stims_[i][0] - time_[i][0],stims_[i][1] - time_[i][0],time_c[anesthetized[0]],time_c[anesthetized[-1]]])

            top_xax.set_xticklabels(["", "", "","Awake"], fontsize=8, style="italic")
            top_xax.tick_params('x', length=2, width=1, which='major')
            top_xax.set_xlim(ax_inset.get_xlim())


            top_xax = ax_.twiny()
            top_xax.set_xticks([stims_[i][0] - time_[i][0],stims_[i][1] - time_[i][0],time_c[anesthetized[0]],time_c[anesthetized[-1]]])

            top_xax.set_xticklabels(["Isoflurane ON", "Isoflurane OFF", "Asleep","Awake"], fontsize=8, style="italic")
            top_xax.tick_params('x', length=2, width=1, which='major')
            top_xax.set_xlim(ax_.get_xlim())

            if idx == len(for_fig) - 1:
                ax_HR_.set_xlabel("Time (s)")
            ax_.set_ylabel("Motion (px/s)")
            ax_HR_.set_ylabel("HR (bpm)")

            max_x = ax_.get_xlim()
            max_y = ax_.get_ylim()

            w = h_inset * np.diff(max_x)
            h = h_inset * np.diff(max_y)

            rect = patches.Rectangle((max_x[1] - w, max_y[1] - h - 102), w, h*1.75, facecolor='white', zorder=99999)
            ax_.add_patch(rect)

            full_x = ax_inset.get_xlim()

            segment_mark = inset_axes(ax_inset, width=1, height=1)
            segment_mark_l = inset_axes(ax_inset, width=1, height=1)
            seg_ip_l = InsetPosition(ax_, [1 - h_inset - spacing/2, 1 - h_inset - spacing * 4 - h_inset/3 - spacing, h_inset* max_x[1]/full_x[1], h_inset/3])

            seg_ip = InsetPosition(ax_, [1 - h_inset - spacing/2, (1 - h_inset - spacing * 4), h_inset* max_x[1]/full_x[1], h_inset])
            segment_mark.set_axes_locator(seg_ip)
            segment_mark_l.set_axes_locator(seg_ip_l)
            #[max_x[0]/full_x[0]-1, 0, max_x[1]/full_x[1], 1]

            segment_mark.set_facecolor((0,0,0,0))
            segment_mark_l.set_facecolor((0,0,0,0))
            for spine in ["bottom", "top", "left", "right"]:
                segment_mark.spines[spine].set_color(rgb_colours[-1])
                segment_mark.spines[spine].set_linewidth(1.2)
                segment_mark_l.spines[spine].set_color(rgb_colours[-1])
                segment_mark_l.spines[spine].set_linewidth(1.2)
                #segment_mark.spines[spine].set_linestyle("-.")

            segment_mark.set_xticks([])
            segment_mark.set_yticks([])
            segment_mark_l.set_xticks([])
            segment_mark_l.set_yticks([])

            # rect = patches.Rectangle((0,0), np.diff(max_x), np.diff(max_y), linewidth=1, edgecolor='r', facecolor='none', zorder=-1)
            # ax_inset.add_patch(rect)

            fig_.tight_layout()
            #axs_[i].set_title(f"{treatm

    ax.axvline(x_ON, linestyle="--",lw=1, color="k") #isoflurane ON
    ax.axvline(x_OFF, linestyle="--",lw=1, color="k") #isoflurane OFF

    ax.set_xticks([stims_[i][0] - time_[i][0],stims_[i][1] - time_[i][0],time_c[anesthetized[0]],time_c[anesthetized[-1]]], minor=True)

    ax.set_xlim(stims_[i][0] - time_[i][0] - offset, stims_[i][0] - time_[i][0] - offset + 150)
    ax.set_ylim(0, 250)


    ax_HR.set_xlim(stims_[i][0] - time_[i][0] - offset, stims_[i][0] - time_[i][0] - offset + 150)
    ax_HR.set_ylim(HR_ll, HR_ul)
    #axs_[i].set_title(f"{treatment}", fontsize=8) #,{weights[included_IDS[i]]}

    # if i == 0:
    #     axs_[0].set_xlabel("time (s)")
    #     axs_[0].set_ylabel("movement speed")
    #
    # else:
    #     axs_[i].set_xticklabels([])
    #     axs_HR[i].set_xticklabels([])
    #     if i % 4 != 0:
    #         axs_[i].set_yticklabels([])
    #         axs_HR[i].set_yticklabels([])



    plt.setp(ax.get_xticklabels(), visible=False)
    fig.add_subplot(ax)
    fig.add_subplot(ax_HR)
    #fig.tight_layout()


marcain = np.array(marcain)
control = np.array(control)
print(stats.ttest_ind(marcain, control,equal_var=False))
print(stats.mannwhitneyu(marcain, control))
print("diff",marcain.mean() - control.mean())
print("percent change:", (marcain.mean()-control.mean())/control.mean())

treatments_ = np.array(treatments_)
onsets = np.array(onsets)

marcain_HRs = np.array(marcain_HRs)
control_HRs = np.array(control_HRs)

#df_ = pd.DataFrame()

df_ = pd.DataFrame()
df_["induction_time"] = pd.cut(onsets, bins = 3) #increase bin number
#print(df_["induction_time"])
df_["treatment"] = treatments_
df_["weight"] = pd.cut(weights_, bins = 3)#np.array(weights_,dtype=int)
df_["age"] = pd.cut(ages_, bins = 3)#np.array(ages_,dtype=int)
df_["gender"] = np.array(genders_)
#print(df_["weight"] )
print(pps.predictors(df_, "induction_time"))



# ind_time = pd.cut(onsets, bins = 3, labels = ["low","medium","high"]) #increase bin number
# treat_ = np.array(treatments_)
# #treat_ = treat_ == "Marcain"
# #treat_ = treat_.astype(int)
#
# weig_ = pd.cut(weights_, bins = 4)#np.array(weights_,dtype=int)#pd.cut(weights_, bins = 4)#
# age_ = np.array(ages_,dtype=int)
# gend_ = np.array(genders_)
# #gend_ = gend_ == "Male"
# #gend_ = gend_.astype(int)
#
# stack = ind_time,treat_, weig_, age_, gend_
# matrix = np.vstack(stack).T
#
# columns=['induction_time','treatment','weight', "age", "gender"]
# df_=pd.DataFrame(matrix,columns=columns)
# df__=pd.DataFrame()
#
# df__["induction_time"] =ind_time
# df__["treatment"] =treat_
# df__["weight"] =weig_
# df__["age"] =age_
# df__["gender"] =gend_
#
corr_ = pd.DataFrame()


columns=['induction_time','treatment','weight', "age", "gender"]
for i, c in enumerate(columns):
    x = pps.predictors(df_, c)["ppscore"]
    #print(c,"----",pps.predictors(df_, c)["ppscore"], "okokokko",pps.predictors(df__, c)["ppscore"])
    x[i + .5 - 1] = 1
    x = x.sort_index().reset_index(drop=True)
    corr_[c] = x

corr_=corr_.rename( index=lambda s: columns[s])
#
print(corr_)
dominance_regression=Dominance(data=corr_,target="induction_time",data_format=1)
incr_variable_rsquare=dominance_regression.incremental_rsquare()
print(incr_variable_rsquare)

fig, axs_bar = plt.subplots(1, 1, figsize =(2.1, 3))
fig.canvas.set_window_title('Induction time comparison')

bplot = axs_bar.boxplot([control, marcain], patch_artist=True, widths=[.4,.4],labels=["Control", "Bupivacaine"], showfliers=False, medianprops = dict(linestyle='-.', linewidth=3, color=rgb_colours[-1]), flierprops = dict(marker='o', markerfacecolor='black', markersize=2,
                  linestyle='none'))

for patch, color in zip(bplot['boxes'], rgb_colours[1::-1]):
    patch.set_facecolor(color)

#axs_bar.bar([0, 1], [np.mean(marcain), np.mean(control)], yerr = [stats.sem(marcain), stats.sem(control)],color=colours[:2], edgecolor="k")
axs_bar.set_ylabel("Induction time (s)")
#axs_bar.set_xticks([0,1])
#axs_bar.set_xticklabels(["Bupivacaine", "Control"])
#axs_bar.set_ylim(32,85)

top_coord = 81
tick_length = 1

axs_bar.plot([1, 1, 2, 2], [top_coord - tick_length, top_coord, top_coord, top_coord - tick_length], linewidth = 1, color='k')

axs_bar.text(1.5, top_coord, '*', style='italic', fontsize=8, ha ="center", in_layout=True)

axs_bar.set_ylim(axs_bar.get_ylim()[0], top_coord + tick_length + 3)
fig.tight_layout()
axs_bar.set_title("A", fontdict=fontdict, loc="left",x=-.23)
axs_bar.set_yticks([40,60,80])

p_perm, ax_perm = permutation_test(marcain, control, plot=True)

if ax_perm != None:
    ax_perm.set_title("B", fontdict=fontdict, loc="left",x=-.23)



fig, axs_box = plt.subplots(1, 1, figsize =(3.5, 3))
fig.canvas.set_window_title('Heartrate comparison')
ibs = .5 #inter box spacing
bplot= axs_box.boxplot(control_HRs, patch_artist=True, positions = [0, 1, 2], widths = 0.6, showfliers = False, medianprops = dict(linestyle='-.', linewidth=3, color=rgb_colours[-1]))

for patch in bplot['boxes']:
    patch.set_facecolor(rgb_colours[1])


bplot=axs_box.boxplot(marcain_HRs, patch_artist=True, positions=[3+ibs,4+ibs,5+ibs],widths = 0.6,showfliers=False, medianprops = dict(linestyle='-.', linewidth=3, color=rgb_colours[-1]))

for patch in bplot['boxes']:
    patch.set_facecolor(rgb_colours[0])

top_xax=axs_box.twiny()
top_xax.set_xlim(axs_box.get_xlim())
top_xax.set_xticks([0,1,2,3+ibs,4+ibs,5+ibs])
top_xax.set_xticklabels(2*["Pre-", "Per-", "Post-"], fontsize=8, style="italic")
top_xax.tick_params('x', length=2, width=1, which='major')

axs_box.set_xticks([1,4+ibs])
axs_box.set_xticklabels(["Control", "Bupivacaine"])
axs_box.set_ylabel("Mean HR (bpm)")
axs_box.set_yticks( [200,240,280])

t, p = stats.ttest_ind(marcain_HRs, control_HRs, axis = 0)
print(f"HR t-test P: {p}")


ext_ttest = f"""
marcain HR, t-test:
pre-per: {stats.wilcoxon(marcain_HRs[:,0],marcain_HRs[:,1])}
per-post: {stats.wilcoxon(marcain_HRs[:,1],marcain_HRs[:,2])}
pre-post: {stats.wilcoxon(marcain_HRs[:,0],marcain_HRs[:,2])}

control HR, P-values:
pre-per: {stats.wilcoxon(control_HRs[:,0], control_HRs[:,1])}
per-post: {stats.wilcoxon(control_HRs[:,1], control_HRs[:,2])}
pre-post: {stats.wilcoxon(control_HRs[:,0], control_HRs[:,2])}
"""

print("okokoko",np.mean(control_HRs[:,0]), np.mean(control_HRs[:,2]))
print("pplplpllp",np.mean(marcain_HRs[:,0]), np.mean(marcain_HRs[:,2]))
print(ext_ttest)

inter_spacing = 8
tick_length = 2
top_coord = HR_ulb + 30

# coords = [[top_coord - inter_spacing * n - tick_length, top_coord - inter_spacing * n,top_coord - inter_spacing * n,top_coord-inter_spacing * n - tick_length] for n in np.arange(1, 4)]
#
# axs_box.plot([0, 0, 3+ibs, 3+ibs], coords[2], linewidth=1, color='k')
# axs_box.plot([1, 1, 4+ibs, 4+ibs], coords[2], linewidth=1, color='k')
# axs_box.plot([2, 2, 5+ibs, 5+ibs], coords[2], linewidth=1, color='k')

#axs_box.plot([0, 0, 3+ibs, 3+ibs], coords[2], linewidth=1, color='k')
#axs_box.plot([1, 1, 4+ibs, 4+ibs], coords[1], linewidth=1, color='k')
#axs_box.plot([2, 2, 5+ibs, 5+ibs], coords[0], linewidth=1, color='k')

#axs_box.text(1.5, coords[2][1], '*', style='italic', font="arial", fontsize=8, ha ="center", in_layout=True)
#axs_box.text(2.5, coords[1][1] + tick_length, 'n.s.', style='italic', font="arial", fontsize=8, ha ="center", in_layout=True)
#axs_box.text(3.5, coords[0][1] + tick_length, 'n.s.', style='italic', font="arial", fontsize=8, ha ="center", in_layout=True)
#axs_box.text(2.5+ibs/2, coords[2][1] + tick_length, 'n.s.', style='italic', font="arial", fontsize=8, ha ="center", in_layout=True)

offs = HR_ulb - 30
diff = 3+ibs

axs_box.text(.5+diff, offs + tick_length, 'n.s.', style='italic', font="arial", fontsize=8, ha ="center", in_layout=True)
axs_box.plot([0+diff, 0+diff, 1+diff, 1+diff], [offs-2, offs, offs,offs-2], linewidth=1, color='k')
x=inter_spacing-20
axs_box.plot([1+diff, 1+diff, 2+diff, 2+diff], [offs-2+x, offs+x, offs+x,offs-2+x], linewidth=1, color='k')
axs_box.text(1.5+diff, offs + tick_length+x, 'n.s.', style='italic', font="arial", fontsize=8, ha ="center", in_layout=True)
x=2*inter_spacing
axs_box.plot([0+diff, 0+diff, 2+diff, 2+diff], [offs-2+x, offs+x, offs+x,offs-2+x], linewidth=1, color='k')
axs_box.text(1+diff, offs + tick_length+x, 'n.s.', style='italic', font="arial", fontsize=8, ha ="center", in_layout=True)

offs2 = HR_ulb

x= -18 - 3
axs_box.text(4.5 + ibs-diff, offs2+x, '**', style='italic', font="arial", fontsize=8, ha ="center", in_layout=True)
axs_box.plot([4 + ibs-diff, 4+ibs-diff, 5+ibs-diff, 5+ibs-diff], [offs2-2+x, offs2+x, offs2+x,offs2-2+x], linewidth=1, color='k')

x+= inter_spacing + 2
axs_box.text(4+ibs-diff, tick_length+offs2+x, 'n.s.', style='italic', font="arial", fontsize=8, ha ="center", in_layout=True)
axs_box.plot([3+ibs-diff, 3+ibs-diff, 5+ibs-diff, 5+ibs-diff], [offs2 - 2+x, offs2+x, offs2+x,offs2-2+x], linewidth=1, color='k')

x=-100
axs_box.plot([3+ibs-diff, 3+ibs-diff, 4+ibs-diff, 4+ibs-diff], [offs2-2+x, offs2+x, offs2+x,offs2-2+x], linewidth=1, color='k')
axs_box.text(3.5+ibs-diff, offs2+x, '**', style='italic', font="arial", fontsize=8, ha ="center", in_layout=True)
#


axs_box.axvline(2.5+ibs/2, linestyle="--",color="grey", alpha=.3, lw=1, zorder=-1)

axs_box.set_ylim(axs_box.get_ylim()[0], 2*tick_length+HR_ulb)#coords[2][1] + inter_spacing)
#axs_box.locator_params(axis='y',nbins=8)
axs_box.set_yticks([160, 220, 280, 340])
fig.tight_layout()

plt.show()

print(f"(success) {TYPE} pipeline complete")
