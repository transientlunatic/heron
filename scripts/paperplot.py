def paper_plot(p, phases=None, approximants=["IMRPhenomPv2", "SEOBNRv4"], matches_flag=True):
    width = 4.5
    height = width/1.618

    f = plt.figure(constrained_layout=True, figsize=(2 * width, height))#, dpi=500)

    gs = GridSpec(1,4, figure = f)

    samples = gp_cat.waveform_samples(p=p,
                                  time_range=[-150, 100, 1024], samples=100)
    mean, variance = gp_cat.waveform(p=p, time_range=[-150,100,1024])
    
    # Set up the approximants
    if "SEOBNRv4" in approximants:
        ts_seo = partial(seo_cat.waveform, p=p, time_range=[-150., 100., 1024])
    if "IMRPhenomPv2" in approximants:
        ts_imr = partial(imr_cat.waveform, p=p, time_range=[-150., 100., 1024])
    
    
    if not phases:
        
        # Calculate optimal matches first
        if "IMRPhenomPv2" in approximants:
            matchimr, phaseimr = optim_match(mean, ts_imr)
        if "SEOBNRv4" in approximants:
            matchseo, phaseseo = optim_match(mean, ts_seo)
            
    else:
        if "IMRPhenomPv2" in approximants:
            phaseimr = phases['IMRPhenomPv2']
            matchimr = match(mean, ts_imr)
        if "SEOBNRv4" in approximants:
            phaseseo = phases['SEOBNRv4']
            matchseo = match(mean, ts_seo)   
        #print phaseimr, phaseseo
        #print matchimr, matchseo
    
    
    # Waveform plot
    times = samples[0].times/1e4 #np.linspace(-150, 100, 1024)/1e4
    #std = np.array(waveforms).std(axis=0)
    ax_wave = f.add_subplot(gs[0:3])
    ax_hist = f.add_subplot(gs[3])
    for sample in samples[1:]:
        ax_wave.plot(sample.times/1e4, sample.data/1e19, color='k', alpha=0.0525, lw=0.5)
    ax_wave.plot(samples[0].times/1e4, samples[0].data/1e19, color='k', alpha=0.0525, lw=0.5, label="GPR Draws")
    ax_wave.plot(times, mean/1e19, label = "GPR Mean", 
                 linestyle="--",
                 alpha=0.5, color='k', lw=2)
    ax_wave.fill_between(times, (mean+variance**2)/1e19, (mean-variance**2)/1e19, alpha=0.1, color='k', label="GPR Variance")

    
    if "IMRPhenomPv2" in approximants:
        # IMRPhenomPv2
        waveform_imr = ts_imr(coa_phase=phaseimr[0], t0=phaseimr[1])[0]

        ax_wave.plot(waveform_imr.times, waveform_imr.data, label="IMRPhenomPv2", 
                     lw=2, alpha=0.8, color="#348ABD")
    if "SEOBNRv4" in approximants:
        # SEOBNRv4
        waveform_seo = ts_imr(coa_phase=phaseseo[0], t0=phaseseo[1])[0]
        ax_wave.plot(waveform_seo.times, waveform_seo.data, label="SEOBNRv4", 
                     lw=2, alpha=0.8, color="#E24A33")
            
    ax_wave.legend(prop=ssp_legend)

    ax_wave.set_xlim([-0.01, 0.01])

    imr_matches = []
    seo_matches = []
    
    if matches_flag:
        for sample in samples:
            if "IMRPhenomPv2" in approximants:
                imr_matches.append(match(sample.data, ts_imr, phase=phaseimr[0], t0=phaseimr[1]))
            if "SEOBNRv4" in approximants:
                seo_matches.append(match(-sample.data, ts_seo, phase=phaseseo[0], t0=phaseseo[1]))

        if "IMRPhenomPv2" in approximants:
            ax_hist.hist(1.0 - np.array(imr_matches), 
                         range=(0,.25), 
                         density=True,
                         bins=21, 
                         histtype="stepfilled", 
                         alpha=0.4,
                         color="#348ABD",
                         label="IMRPhenomPv2"
                        )

        if "SEOBNRv4" in approximants:
            ax_hist.hist(1.0 - np.array(seo_matches), 
                     range=(0,.25), 
                     density=True,
                     bins=21, 
                     histtype="stepfilled", 
                     alpha=0.4,
                     color="#E24A33",
                     label="SEOBNRv4"
                    )
        ax_hist.legend(prop=ssp_legend)
        #ax_hist.vlines(x=[1-mean_match_imr, 1-mean_match_seo], ymax=35, ymin=0,
        #              color=['blue', 'red'])

        if "IMRPhenomPv2" in approximants:
            ax_hist.vlines(1-matchimr, *ax_hist.get_ylim(), color="#348ABD")
        if "SEOBNRv4" in approximants:
            ax_hist.vlines(1-matchseo, *ax_hist.get_ylim(), color="#E24A33")
        ax_wave.set_xlabel("Time since merger [s]", fontdict=lato)
        ax_hist.set_xlabel("Mismatch", fontdict=lato)
        ax_wave.set_ylabel("Strain", fontdict=lato)
    
    for label in ax_wave.get_xticklabels():
        label.set_fontproperties(ticks_font)
    for label in ax_wave.get_yticklabels():
        label.set_fontproperties(ticks_font)
    for label in ax_hist.get_xticklabels():
        label.set_fontproperties(ticks_font)
    for label in ax_hist.get_yticklabels():
        label.set_fontproperties(ticks_font)
    ax_hist.xaxis.get_offset_text().set_fontproperties(ticks_font)
    
    ax_hist.yaxis.get_offset_text().set_fontproperties(ticks_font)
    ax_wave.xaxis.get_offset_text().set_fontproperties(ticks_font)
    
    ax_wave.yaxis.get_offset_text().set_fontproperties(ticks_font)
    #f.tight_layout()
    return f, [ax_wave, ax_hist], #{"imr": {"phase": phaseimr}, "seo": {"phase": phaseseo}}
