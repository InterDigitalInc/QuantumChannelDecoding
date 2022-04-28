# **********************************************************************************************************************
# The functions in this file facilitate running experiments with different parameters and displaying the experiment
# results for Hamming codes.
#
# This is part of the code for the paper "Quantum Channel Decoding" as submitted to the "QEC-22" conference:
#    https://qce.quantum.ieee.org/2022/
#
# Copyright (c) 2022 InterDigital AI Research Lab
# Author: Shahab Hamidi-Rad
# **********************************************************************************************************************
import numpy as np
import time, sys, yaml
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['figure.dpi'] = 100

from qiskit.visualization import plot_histogram
from quantumhammingdecoder import QuantumHamming

# **********************************************************************************************************************
# The format of the results in the Yaml files:
# For Quantum:  code->decoder->sim->errortype->array
# For others:   code->decoder->errortype->array
# **********************************************************************************************************************

# **********************************************************************************************************************
# myPrint: Utility function for printing in the interactive mode
def myPrint(textStr, eol=True):
    if eol == False:
        sys.stdout.write(textStr)
        sys.stdout.flush()
    else:
        sys.stdout.write(textStr+'\n')
        sys.stdout.flush()
        
# **********************************************************************************************************************
# saveYaml: Save the results to a Yaml file.
def saveYaml(fileName, data):
    myPrint("Saving to '%s'..."%(fileName), False)
    with open(fileName, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    myPrint("Done.")

# **********************************************************************************************************************
# loadYaml: Load the saved results from a Yaml file.
def loadYaml(fileName):
    with open(fileName, 'r') as infile:
        try:
            return yaml.safe_load(infile)
        except yaml.YAMLError as err:
            print(err)
            return None

# **********************************************************************************************************************
# printInfo: prints general information about a results file dictionary.
def printInfo(exp):
    decodersKey = "decoders" if "decoders" in exp['params'] else "testCases"
    print("\nFile Version:        ",str(exp['version']))
    print("Decoding Algorithms: ","   ".join(exp['params'][decodersKey]))
    print("Number of Blocks:    ", str(exp['params']['numBlocks']))
    print("Number of Shots:     ", str(exp['params']['numShots']))
    if 'sigma2' in exp['params']:
        if exp['params']['sigma2'] is not None:
            print("Sigma2 fixed to:     ",exp['params']['sigma2'])
    print("Eb/No Values(db):    ","   ".join(str(d) for d in exp['params']['ebNoDbs']))
    print("Simulators:          ","   ".join(exp['params']['simulators']))
            
    for c,codeStr in enumerate(exp['results'].keys()):
        print("Hamming %s Times:\n  Classical: %.2f Sec.\n  Quantum: %.2f Sec."%(codeStr,
                                                                                 exp['classicTimesPerCode'][c],
                                                                                 exp['quantumTimesPerCode'][c]))

# **********************************************************************************************************************
# printResults: prints the results using experiment information in the "exp" dictionary. Other parameters can be
# used to filter out the information and print only the results for the specified parameter values.
def printResults(exp, codeStr=None, simulator=None, ber=False, decoders=None, fileInfo=True):
    if fileInfo: printInfo(exp)
    bfKey = 'bers' if ber else 'fers'
    params = exp["params"]
    results = exp["results"]
    assert exp["version"]==4, "Cannot handle version '%d' data! Only version 4 is supported by this function."%(exp["version"])
    decodersKey = "decoders" if "decoders" in params else "testCases"
    decoders = params[decodersKey] if decoders is None else [x for x in params[decodersKey] if x in decoders]
    for n,k in params["codes"]:
        cStr = "(%d,%d)"%(n,k)
        if (codeStr is not None) and (cStr != codeStr): continue
        for sim in params["simulators"]:
            if (simulator is not None) and (simulator != sim): continue
            print("\n%s for %s Hamming Code, %s"%(bfKey.upper()[:3], cStr, sim))

            print("Eb/No(db)  " + ''.join(("%%-%ds  "%(max(len(x),9)))%(x) for x in decoders))
            for i,ebNoDb in enumerate(params["ebNoDbs"]):
                rowStr = "%-9d  "%(ebNoDb)
                for decoder in decoders:
                    if decoder in  ["Quantum", "Quantum-Fast", "Quantum-Corr"]:
                        rowStr += ("%%-%d.6f  "%(max(len(decoder),9)))%(results[cStr][decoder][sim][bfKey][i])
                    else:
                        rowStr += ("%%-%d.6f  "%(max(len(decoder),9)))%(results[cStr][decoder][bfKey][i])
                print(rowStr)

# **********************************************************************************************************************
# drawResults: Draws graph(s) based on the experiment information in the "exp" dictionary. Other parameters can be
# used to filter out the information and draw only the graphs for the specified parameter values.
def drawResults(exp, codeStr=None, simulator=None, ber=False, decoders=None, title=None, logY=True):
    bfKey = 'bers' if ber else 'fers'
    mrk = 'o^+xDvsp'    # Markers
    ls = 3              # Line width
    
    params = exp["params"]
    results = exp["results"]
    assert exp["version"]==4, "Cannot handle version '%d' data! Only version 4 is supported by this function."%(exp["version"])
    decodersKey = "decoders" if "decoders" in params else "testCases"
    decoders = params[decodersKey] if decoders is None else [x for x in params[decodersKey] if x in decoders]
    for n,k in params["codes"]:
        cStr = "(%d,%d)"%(n,k)
        if (codeStr is not None) and (cStr != codeStr): continue
        for sim in params["simulators"]:
            if (simulator is not None) and (simulator != sim): continue
            fig, ax = plt.subplots()
            if title is None:
                ax.set_title("%s at different $E_b/N_0$ values for %s Hamming Code\n(%s)"%(bfKey.upper()[:3], cStr, sim), fontsize=20)
            elif title!="":
                ax.set_title(title, fontsize=20)

            for i,decoder in enumerate(decoders):
                if decoder in ["Quantum", "Quantum-Fast", "Quantum-Corr"]:
                    ax.plot(params["ebNoDbs"], results[cStr][decoder][sim][bfKey],
                            label=decoder, marker=mrk[i], markersize=3*ls, linewidth=3)
                else:
                    labels = {"Hard":"HD", "Soft":"ML", "No Coding":"No Coding"}
                    ax.plot(params["ebNoDbs"], results[cStr][decoder][bfKey],
                            label=labels[decoder], marker=mrk[i], markersize=3*ls, linewidth=3)

            ax.legend(prop={'size': 32})
            if logY: ax.set_yscale('log')
            plt.xlabel(r'$E_b/N_0$ (db)', fontsize=36)
            plt.ylabel(bfKey.upper()[:3], fontsize=36)
            plt.xticks(params["ebNoDbs"], fontsize=32)
            plt.yticks(fontsize=32)
            plt.grid()

# **********************************************************************************************************************
# runExperiments: Runs experiments based on the provided parameters. It loops through all combinations of the given
# parameters and performs the experiments. The results is then saved to the specified Yaml file.
# NOTE: Depending on the parameters given and your computer power, this can take hours or even days to complete.
def runExperiments(fileName, **kwargs):
    nBlocks =    kwargs.get("nBlocks", 10000)                           # Number of blocks to try for each combination
    nShots =     kwargs.get("nShots", 1024)                             # Number of shots for quantum circuit execution
    codes =      kwargs.get("codes", [[7,4], [15,11]])                  # Coding configurations in the form of [n,k,nCrc]
    ebNoDbs =    kwargs.get("ebNoDbs", list(range(-4,9,2)))             # The Eb/N0 values used for the experiments.
    decoders =   kwargs.get("decoders",
                            ["No Coding", "Quantum", "Hard", "Soft"])   # The decoding algorithms
    simulators = kwargs.get("simulators", [                             # Qiskit simulators to use. If you have a GPU
#        "aer_simulator",                                               # machine, use "aer_simulator_statevector_gpu".
        "aer_simulator_statevector",
#        "aer_simulator_statevector_gpu",
#        "aer_simulator_density_matrix",       # This crashes
#        "aer_simulator_stabilizer",           # This complains that some gates are not supported.
#        "aer_simulator_matrix_product_state", # Kernel dies!
    ])
    fixedSigma2 =    kwargs.get("sigma2", None)

    # YAML does not like tuples!
    codes = [list(x) for x in codes]
    
    errorRates = {
                    "version":              4,
                    "params": {
                        "codes":            codes,
                        "numBlocks":        nBlocks,
                        "numShots":         nShots,
                        "ebNoDbs":          list(ebNoDbs),
                        "decoders":         decoders,
                        "simulators":       simulators,
                        "sigma2":           fixedSigma2
                    },
                    "classicTimesPerCode":  [], # One per code in "codes"
                    "quantumTimesPerCode":  [], # One per code in "codes"
                    "totalTime":            0,
                    "results":              {}
                 }

    results = {}
    for n,k in codes:
        codeStr = "(%d,%d)"%(n,k)
        results[codeStr] = {}
        for decoder in decoders:
            if decoder in ["Quantum", "Quantum-Fast", "Quantum-Corr"]:
                # code->decoder->sim->errortype->array
                results[codeStr][decoder] = {s:{"bers":[], "fers":[], "maxCountsLen":0} for s in simulators}
            else:
                # code->decoder->errortype->array
                results[codeStr][decoder] = {"bers":[], "fers":[]}

    t0 = time.time()
    for n,k in codes:
        codeStr = "(%d,%d)"%(n,k)
        t00 = time.time()
        
        # Use the same set of messages for all simulators and classical methods
        qhTemp = QuantumHamming(n)
        msg, codeWord, tx, rx, sigma2 = qhTemp.getRandomMessage(ebNoDbs, nBlocks)

        for s,sim in enumerate(simulators):
            qh = QuantumHamming(n, simulator=sim)
            if s==0:
                if "No Coding" in decoders:
                    t000 = time.time()
                    myPrint("No Coding (Message Len=%d) ... "%(k), False)
                    # No Coding: (Rate=1.0)
                    sigmaNC = 1/np.sqrt(2*(10.0**(np.array(ebNoDbs)/10.0)))
                    rxsNoCode = np.random.normal(1 - 2.0*msg, np.expand_dims(sigmaNC,(1,2)))
                    msgHats = np.int8(rxsNoCode<0)  # Hard Decision
                    bitErrors = (msgHats^msg).sum(axis=2)
                    frameErrors = bitErrors>0
                    
                    results[codeStr]["No Coding"]["bers"] = (bitErrors.sum(axis=1)/(nBlocks*k)).tolist()
                    results[codeStr]["No Coding"]["fers"] = (frameErrors.sum(axis=1)/nBlocks).tolist()
                    
                    myPrint("Done (%.02f Sec.)"%(time.time()-t000))

                if "Hard" in decoders:
                    t000 = time.time()
                    myPrint("Hamming Decoding %s, Hard Decision ... "%(codeStr), False)
                    msgHats = np.int8([[qh.hardDecode(rx[e][m]) for m in range(nBlocks)] for e in range(len(ebNoDbs))])
                    bitErrors = (msgHats^msg).sum(axis=2)
                    frameErrors = bitErrors>0
                    results[codeStr]["Hard"]["bers"] = (bitErrors.sum(axis=1)/(nBlocks*k)).tolist()
                    results[codeStr]["Hard"]["fers"] = (frameErrors.sum(axis=1)/nBlocks).tolist()
                    myPrint("Done (%.02f Sec.)"%(time.time()-t000))

                if "Soft" in decoders:
                    t000 = time.time()
                    myPrint("Hamming Decoding %s, Maximum Likelihood ... "%(codeStr), False)
                    msgHats = np.int8([[qh.softDecode(rx[e][m]) for m in range(nBlocks)] for e in range(len(ebNoDbs))])
                    bitErrors = (msgHats^msg).sum(axis=2)
                    frameErrors = bitErrors>0
                    results[codeStr]["Soft"]["bers"] = (bitErrors.sum(axis=1)/(nBlocks*k)).tolist()
                    results[codeStr]["Soft"]["fers"] = (frameErrors.sum(axis=1)/nBlocks).tolist()
                    myPrint("Done (%.02f Sec.)"%(time.time()-t000))

                errorRates["classicTimesPerCode"] += [time.time()-t00]
                t00 = time.time() # From now on everything is quantum

            if "Quantum" in decoders:
                t000 = time.time()
                myPrint("Hamming Decoding %s, Quantum (%s) ... "%(codeStr, sim), False)
                if fixedSigma2 is None:
                    msgHats = np.int8([[qh.decode(rx[e][m], sigma2[e], numShots=nShots) for m in range(nBlocks)] for e in range(len(ebNoDbs))])
                else:
                    msgHats = np.int8([[qh.decode(rx[e][m], fixedSigma2, numShots=nShots) for m in range(nBlocks)] for e in range(len(ebNoDbs))])
                        
                bitErrors = (msgHats^msg).sum(axis=2)
                frameErrors = bitErrors>0
                results[codeStr]["Quantum"][sim]["bers"] = (bitErrors.sum(axis=1)/(nBlocks*k)).tolist()
                results[codeStr]["Quantum"][sim]["fers"] = (frameErrors.sum(axis=1)/nBlocks).tolist()
                myPrint("Done (%.02f Sec.)"%(time.time()-t000))

            if "Quantum-Fast" in decoders:
                t000 = time.time()
                myPrint("Hamming Decoding %s, Quantum-Fast (%s) ... "%(codeStr, sim), False)
                if fixedSigma2 is None:
                    msgHats = np.int8([[qh.decodeFast(rx[e][m], sigma2[e], numShots=nShots) for m in range(nBlocks)] for e in range(len(ebNoDbs))])
                else:
                    msgHats = np.int8([[qh.decodeFast(rx[e][m], fixedSigma2, numShots=nShots) for m in range(nBlocks)] for e in range(len(ebNoDbs))])
                        
                bitErrors = (msgHats^msg).sum(axis=2)
                frameErrors = bitErrors>0
                results[codeStr]["Quantum-Fast"][sim]["bers"] = (bitErrors.sum(axis=1)/(nBlocks*k)).tolist()
                results[codeStr]["Quantum-Fast"][sim]["fers"] = (frameErrors.sum(axis=1)/nBlocks).tolist()
                myPrint("Done (%.02f Sec.)"%(time.time()-t000))

            if "Quantum-Corr" in decoders:
                t000 = time.time()
                myPrint("Hamming Decoding %s, Quantum-Corr (%s) ... "%(codeStr, sim), False)
                if fixedSigma2 is None:
                    msgHats = np.int8([[qh.decode(rx[e][m], sigma2[e], numShots=nShots, correctErrors=True) for m in range(nBlocks)] for e in range(len(ebNoDbs))])
                else:
                    msgHats = np.int8([[qh.decode(rx[e][m], fixedSigma2, numShots=nShots, correctErrors=True) for m in range(nBlocks)] for e in range(len(ebNoDbs))])
                        
                bitErrors = (msgHats^msg).sum(axis=2)
                frameErrors = bitErrors>0
                results[codeStr]["Quantum-Corr"][sim]["bers"] = (bitErrors.sum(axis=1)/(nBlocks*k)).tolist()
                results[codeStr]["Quantum-Corr"][sim]["fers"] = (frameErrors.sum(axis=1)/nBlocks).tolist()
                myPrint("Done (%.02f Sec.)"%(time.time()-t000))

        errorRates["quantumTimesPerCode"] += [time.time()-t00]

    errorRates["totalTime"] = time.time()-t0
    errorRates["results"] = results
    myPrint("Total Time: %.02f Sec."%(errorRates["totalTime"]))

    saveYaml(fileName, errorRates)
    return errorRates
