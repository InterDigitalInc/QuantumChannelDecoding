# **********************************************************************************************************************
# This file implements the "QuantumHamming" class which is a Quantum Hamming Decoder. It uses the "Quantum Soft
# Decision" and "Quantum Generator" concepts to build a quantum circuit and decode the received noisy signals and
# recover the original messages.
#
# This is part of the code for the paper "Quantum Channel Decoding" as submitted to the "QEC-22" conference:
#    https://qce.quantum.ieee.org/2022/
#
# Copyright (c) 2022 InterDigital AI Research Lab
# Author: Shahab Hamidi-Rad
# **********************************************************************************************************************
import numpy as np
import matplotlib.pyplot as plt

# Note: If both "Qiskit" and "Amazon Braket" are installed, by default we use "Qiskit". In this case, if you want to
# use "Amazon Braket", you can use: QuantumPolar.setPlatform("BRAKET")
PLATFORM=None
try:
    # AWS imports: Import Braket SDK modules
    from braket.circuits import Circuit, Gate, Observable
    from braket.devices import LocalSimulator
    from braket.aws import AwsDevice
    PLATFORM="BRAKET"
except:
    pass

try:
    # Qiskit Imports:
    from qiskit import QuantumCircuit, Aer, execute, transpile
    from qiskit.providers.aer import AerSimulator
    import qiskit.test.mock as fakes
    PLATFORM="QISKIT"
except:
    pass
assert PLATFORM is not None, "You need to install at least one of 'Qiskit' or 'Amazon Braket' packages!"

# **********************************************************************************************************************
def bs2Str(bs):     return "".join(str(x) for x in bs)
def str2bs(bsStr):  return np.int8([int(x) for x in bsStr])
def bs2Int(bs,l):   return (bs*[1<<x for x in range(l-1,-1,-1)]).sum()-1

# **********************************************************************************************************************
# QuantumHamming: This class encapsulates the functionality to decode hamming codes using a quantum circuit.
class QuantumHamming:
    def __init__(self, n, **kwargs):
        self.counts = None
        self.circuit = None
        self._backend = None
        self.fakeComputer = None
        self.device = kwargs.get("device", None)
        if self.device is None and PLATFORM=="BRAKET": self.device = LocalSimulator()
        self.setSimulator(kwargs.get("simulator", "aer_simulator"))

        self.n = n
        numParity = int(np.log2(n)+1)
        self.k = n - numParity
    
        parityCoverage = []
        for i in range(numParity):
            parityCoverage += [ [ 1 if ((1<<i)&(d+1))!=0 else 0 for d in range(n) if d&(d+1) ] ]
        parityCoverage = np.int8(parityCoverage)

        self.g = np.int8(np.zeros((self.k,self.n)))
        self.pIdx = [p for p in range(n) if p&(p+1)==0]
        self.dIdx = [d for d in range(n) if d&(d+1)]
        self.g[:,self.pIdx] = parityCoverage.T
        self.g[:,self.dIdx] = np.eye(self.k)
            
        self.h = np.int8(np.zeros((numParity,n)))
        self.h[:,self.dIdx] = parityCoverage
        self.h[:,self.pIdx] = np.eye(numParity)

        self.rate = self.k/self.n

    # ******************************************************************************************************************
    @classmethod
    def setPlatform(cls,platform):
        global PLATFORM
        if platform != PLATFORM:
            print("Software platform changed to '%s'."%(platform))
        PLATFORM = platform

    # ******************************************************************************************************************
    @classmethod
    def getHammingGH(clse, n):
        # Source: Wikipedia -> https://en.wikipedia.org/wiki/Hamming(7,4)
        numParity = int(np.log2(n)+1)
        k = n - numParity
        
        parityCoverage = []
        for i in range(numParity):
            parityCoverage += [ [ 1 if ((1<<i)&(d+1))!=0 else 0 for d in range(n) if d&(d+1) ] ]
        parityCoverage = np.int8(parityCoverage)
        
        g = np.int8(np.zeros((k,n)))
        pIdx = [p for p in range(n) if p&(p+1)==0]
        dIdx = [d for d in range(n) if d&(d+1)]
        g[:,pIdx] = parityCoverage.T
        g[:,dIdx] = np.eye(k)
        
        h = np.int8(np.zeros((n-k,n)))
        h[:,dIdx] = parityCoverage
        h[:,pIdx] = np.eye(n-k)

        return g,h

    # ******************************************************************************************************************
    def getValidCodeWords(self):
        # return a list of all valid codewords:
        return np.int8([ str2bs(("{0:0%db}"%(self.k)).format(m)).dot(self.g)%2 for m in range(1<<self.k) ])
        
    # ******************************************************************************************************************
    def hardDecode(self, rx, minDist=True):
        cwHard = np.int8(rx<0)
        if minDist:
            validCodewords = self.getValidCodeWords()
            dists = ((cwHard+validCodewords)%2).sum(axis=1)
            cwHatIdx = np.argmin(dists)
            cwHat = validCodewords[ cwHatIdx ]
            msg = cwHat[self.dIdx]
            return msg

        errBits = (self.h.dot(cwHard)%2)[::-1]
        errIdx = (errBits*[1<<x for x in range(self.n-self.k-1,-1,-1)]).sum()-1
        if errIdx>=0:
            cwHard[errIdx] = 1-cwHard[errIdx]  # Flip the error bit
        msg = cwHard[self.dIdx]
        return msg

    # ******************************************************************************************************************
    def softDecode(self, rx):
        validCodewords = self.getValidCodeWords()
        # Calculate the correlations with all valid codewords:
        corr = (rx*(1-2.0*validCodewords)).sum(axis=1)
        cwHatIdx = np.argmax(corr)
        cwHat = validCodewords[ cwHatIdx ]
        msg = cwHat[self.dIdx]
        return msg

    # ******************************************************************************************************************
    @property
    def backend(self):
        if self._backend is None:
            self._backend = Aer.get_backend(self.simulator)
            if self.device is not None: self._backend.set_options(device=self.device)
        return self._backend
        
    # ******************************************************************************************************************
    def setSimulator(self, sim):
        if sim[:4].lower()=="fake":
            # For more info about "Fake" Quantum Computers visit:
            #   https://qiskit.org/documentation/tutorials/simulators/2_device_noise_simulation.html
            assert PLATFORM=="QISKIT", "Fake Quantum Computers are only supported with QISKIT software platform!"
            fakeComputerClass = getattr(fakes, sim)
            fakeBackend = fakeComputerClass()
            self.fakeComputer = AerSimulator.from_backend(fakeBackend)
            self._backend = fakeBackend
            self.simulator = sim
        else:
            self.simulator = sim
            self._backend = Aer.get_backend(self.simulator)
            if self.device is not None: self._backend.set_options(device=self.device)
        
    # ******************************************************************************************************************
    def getRandomMessage(self, ebNoDb=4, count=1):
        msgs = np.random.randint(0,2,(count,self.k))
        
        codeWords = msgs.dot(self.g)%2
        tBpsk = 1 - 2.0*codeWords
        
        if type(ebNoDb) in [list, np.ndarray]:  ebNoDb = np.array(ebNoDb)
        else:                                   ebNoDb = np.array([ebNoDb])
        ebNo = 10.0**(ebNoDb/10)
        sigma = 1/np.sqrt(2*self.rate*ebNo)
        sigma2 = sigma*sigma
        
        rx = np.random.normal(tBpsk, np.expand_dims(sigma,(1,2)))
        if len(ebNoDb) == 1: rx,sigma2 = rx[0],sigma2[0]

        if count == 1:
            return msgs[0], codeWords[0], tBpsk[0], np.squeeze(rx), sigma2
        return msgs, codeWords, tBpsk, rx, sigma2
    
    # ******************************************************************************************************************
    def qbsStr2bsStr(self, qbs):
        return qbs[::-1] if PLATFORM=="QISKIT" else qbs

    # ******************************************************************************************************************
    def qbs2msgErrStr(self, qbs):
        bs = qbs[::-1] if PLATFORM=="QISKIT" else qbs
        msgStr = "".join(bs[i] for i in range(self.n) if i in self.dIdx)
        errStr = "".join(bs[i] for i in range(self.n) if i in self.pIdx)
        return msgStr+"-"+errStr

    # ******************************************************************************************************************
    def getBsCounts(self):
        # The bitstream we get from the circuit contains only message bits. We now convert them to uHat bitstreams
        # by inserting frozen bits and reordering the message bits.
        bsCounts = {}
        for bs, count in self.counts.items():
            msgErrStr = self.qbs2msgErrStr(bs)
            bsCounts[msgErrStr] = bsCounts.get(msgErrStr,0) + count
        return bsCounts
    
    # ******************************************************************************************************************
    def getTopKBitstream(self, k=None):
        topK = sorted(self.counts.items(), key=lambda x: -x[1])
        if k is not None: topK = topK[:k]
            
        topKInfo = []
        for bsStr, count in topK:
            bs = str2bs(bsStr)[::-1] if PLATFORM=="QISKIT" else str2bs(bsStr)
            errBits = bs[self.pIdx]
            errIdx = (errBits*[1<<x for x in range(self.n-self.k-1,-1,-1)]).sum()-1
            if errIdx>=0: bs[errIdx] = 1-bs[errIdx]
            msgHat = bs[self.dIdx]
            cwHat = msgHat.dot(self.g)%2
            topKInfo += [(bs, count, self.qbs2msgErrStr(bsStr), msgHat, cwHat)]

        return topKInfo

    # ******************************************************************************************************************
    def getQuantumGen(self):
        cNotGates = []
        for r,row in enumerate(self.h):
            target = -1
            for c,col in enumerate(row):
                if col == 0: continue
                if target==-1: target = c
                else:          cNotGates += [(c,target)]
                        
        return cNotGates
        
    # ******************************************************************************************************************
    def buildCircuit(self, rx, sigma2):
        cNotGates = self.getQuantumGen()
        hammingCirc = QuantumCircuit(self.n) if PLATFORM=="QISKIT" else Circuit()
        self.numQubits = self.n
        
        # Quantum Soft Decision
        for q in range(self.n):
            # Initialize based on the probabilities:
            prob1 = 1/(1+np.exp(2*rx[q]/sigma2))    # Probability of i'th bit being 1
            theta = np.arcsin(np.sqrt(prob1))*2     # Projection onto Z axis gives probability magnitude
            
            # Rotate based on probability => Soft decision
            if PLATFORM=="QISKIT":  hammingCirc.ry(theta, q)
            else:                   hammingCirc.ry(q, theta)

        if PLATFORM=="QISKIT":      hammingCirc.barrier()

        # Quantum Generator
        for q1, q2 in cNotGates:
            if PLATFORM=="QISKIT":  hammingCirc.cx(q1, q2)
            else:                   hammingCirc.cnot(control=q1, target=q2)

        if PLATFORM=="QISKIT":      hammingCirc.measure_all()
        return hammingCirc

    # ******************************************************************************************************************
    def getProb(self, cw, rx, sigma2):
        prob = 1
        for i in range(self.n):
            probEq1 = 1/(1+np.exp(2*rx[i]/sigma2))
            probEqCw = probEq1 if cw[i] else (1.0-probEq1)
            prob *= probEqCw
        return prob

    # ******************************************************************************************************************
    def run(self, numShots):
        if PLATFORM=="QISKIT":
            if self.fakeComputer is not None:
                tcirc = transpile(self.circuit, self.fakeComputer)
                return self.backend.run(tcirc, seed_simulator=10, shots=numShots).result().get_counts()

            return self.backend.run(self.circuit, seed_simulator=10, shots=numShots).result().get_counts()
        
        return self.device.run(self.circuit, shots=numShots).result().measurement_counts

    # ******************************************************************************************************************
    def decodeFast(self, rx, sigma2=0.5, numShots=1024, topK=None, executeCircut=True):
        if executeCircut:
            self.circuit = self.buildCircuit(rx, sigma2)
            self.counts = self.run(numShots)

        self.topKcounts = sorted(self.counts.items(), key=lambda x: -x[1])
        if topK is not None: self.topKcounts = self.topKcounts[:topK]

        bsStr = self.qbsStr2bsStr(self.topKcounts[0][0])
        msgPcheck = str2bs(bsStr)
        msgHat = msgPcheck[self.dIdx]

        # Do the correction
        pCheck = msgPcheck[self.pIdx]
        errIdx = (pCheck*[1<<x for x in range(self.n-self.k-1,-1,-1)]).sum()-1
        msgHatCor = (np.int8(self.dIdx)==errIdx)^msgHat
        return msgHatCor

    # ******************************************************************************************************************
    def decode(self, rx, sigma2=0.5, numShots=1024, topK=None, correctErrors=False, executeCircut=True):
        if executeCircut:
            self.circuit = self.buildCircuit(rx, sigma2)
            self.counts = self.run(numShots)

        self.topKcounts = sorted(self.counts.items(), key=lambda x: -x[1])
        if topK is not None: self.topKcounts = self.topKcounts[:topK]

        topKmsgs = np.int8([ str2bs(self.qbsStr2bsStr(qbsStr))[self.dIdx] for qbsStr,_ in self.topKcounts ])
        topKcws = topKmsgs.dot(self.g)%2
        correlations = (rx*(1-2.0*topKcws)).sum(axis=1)
        idx = np.argmax(correlations)
        
        if correctErrors:
            topKpChecks = np.int8([ str2bs(self.qbsStr2bsStr(qbsStr))[self.pIdx] for qbsStr,_ in self.topKcounts ])
            p2s = np.int8([1<<x for x in range(self.n-self.k-1,-1,-1)])
            errIdx = (topKpChecks*p2s).sum(1)-1
            topKmsgCors = ( np.int8(len(errIdx)*[self.dIdx]) == errIdx.reshape((-1,1)))^topKmsgs
            topKcwCors = topKmsgCors.dot(self.g)%2
            correlationCors = (rx*(1-2.0*topKcwCors)).sum(axis=1)
            idxCor = np.argmax(correlations)
            if correlationCors[idxCor]>correlations[idx]:
                return topKmsgCors[idxCor]
    
        return topKmsgs[ idx ]

    # ******************************************************************************************************************
    def printStats(self, rx, msgDcd, msgGt, k=None, histogram=True):
        topKCounts = self.topKcounts if k is None else self.topKcounts[:k]

        if k is None:   print("\nAll Candidates:")
        else:           print("\nTop-%d Candidates:"%(k))
                                    # Columns         Desc.
        wB = max(6,self.numQubits)  # Qubits          The measured qubits
        wM = max(3,self.k)          # MSG             The Message Part
        wCw = max(9,self.n)         # Code-Word       The code-word from the message
        wCrl = 11                   # Correlation     The correlation between rx and the code-word from the message

        wC = max(5,self.n-self.k)   # Check           The parity check bits
        wCwR = max(7,self.n)        # CW (rx)         The received codeword (Just after Quantum soft decision)
        wCwC = max(8,self.n)        # CW (cor)        The code-word from the corrected message
        wMc = max(9,self.k)         # MSG (cor)       The message corrected based on the parity check bits
        wCrlC = 11                  # Corrl (cor)     The correlation between rx and "CW (cor)"

        mcStr = "".join("M" if i in self.dIdx else "C" for i in range(self.n))
        mpStr = "".join("M" if i in self.dIdx else "P" for i in range(self.n))
        headerFormat = "    %%-%ds  %%5s  %%-%ds  %%-%ds  %%-%ds  %%-%ds  %%-%ds  %%-%ds  %%-%ds  %%-%ds"%(wB, wM, wCw, wCrl, wC, wCwR, wCwC, wMc, wCrlC)
        print(headerFormat%("Qubits", "Count", "MSG", "Code-Word", "Correlation", "Check", "CW (rx)", "CW (cor)", "MSG (cor)", "Corrl (cor)"))
        print(headerFormat%(mpStr, "", "", mcStr, "", "", mcStr, mcStr, "", ""))

        for qbsStr,count in topKCounts:
            bsStr = self.qbsStr2bsStr(qbsStr)
            msgPcheck = str2bs(bsStr)
            msgHat = msgPcheck[self.dIdx]
            cwHat = msgHat.dot(self.g)%2
            correlation = (rx*(1-2.0*cwHat)).sum()

            # Undoing the Quantum Generator to get CW based on received signal
            cwRx = msgPcheck.copy()
            cwRx[self.pIdx] = self.h.dot(msgPcheck)%2

            # Do the correction
            pCheck = msgPcheck[self.pIdx]
            errIdx = (pCheck*[1<<x for x in range(self.n-self.k-1,-1,-1)]).sum()-1
            msgHatCor = (np.int8(self.dIdx)==errIdx)^msgHat
            cwHatCor = msgHatCor.dot(self.g)%2
            correlationCor = (rx*(1-2.0*cwHatCor)).sum()

            if np.all(msgHat==msgDcd):
                if np.all(msgHat==msgGt):       flag = '✓'
                else:                           flag = "D"
            elif np.all(msgHat==msgGt):         flag = "G"
            else:                               flag = "-"
            
            if np.all(msgHatCor==msgDcd):
                if np.all(msgHatCor==msgGt):    flag += '✓'
                else:                           flag += "D"
            elif np.all(msgHatCor==msgGt):      flag += "G"
            else:                               flag += "-"

            formatStr = "    %%-%ds  %%-5d  %%-%ds  %%-%ds  %%-%d.4f  %%-%ds  %%-%ds  %%-%ds  %%-%ds  %%-%d.4f  %%s"%(wB, wM, wCw, wCrl, wC, wCwR, wCwC, wMc, wCrlC)
            print(formatStr%(bsStr, count, bs2Str(msgHat), bs2Str(cwHat), correlation,
                             bs2Str(pCheck), bs2Str(cwRx), bs2Str(cwHatCor), bs2Str(msgHatCor), correlationCor, flag))

        if histogram:
            # Finally show a histogram of the bitstreams as measured at the output of the circuit
            bsStrs,counts = zip(*topKCounts)
            plt.bar([self.qbsStr2bsStr(bsStr) for bsStr in bsStrs], list(counts))
            plt.xticks(rotation=90)
            plt.xlabel('bit-stream')
            plt.ylabel('counts')
            plt.show()
