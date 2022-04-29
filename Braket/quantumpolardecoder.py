# **********************************************************************************************************************
# This file implements the "QuantumPolar" class which is a Quantum Polar Decoder. It uses the "Quantum Soft Decision"
# and "Quantum Polar Generator" concepts to build a quantum circuit and decode the received noisy signals to recover
# the original messages.
#
# This is part of the code for the paper "Quantum Polar Decoding" as submitted to the "QEC-22" conference:
#    https://qce.quantum.ieee.org/2022/
#
# Copyright (c) 2022 InterDigital AI Research Lab
# Author: Shahab Hamidi-Rad
# **********************************************************************************************************************
import numpy as np
import matplotlib.pyplot as plt
from polarsc import getReliabilitySequence, scDecode, sclDecode

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
def bs2str(bs):     return "".join(str(x) for x in bs)
def str2bs(bsStr):  return np.int8([int(x) for x in bsStr])
def bs2Int(bs,l):   return (bs*[1<<x for x in range(l-1,-1,-1)]).sum()-1

# **********************************************************************************************************************
# CRC functions:
# **********************************************************************************************************************
# The polynomials to use for different number of CRC bits:
crcPolys = [None,
            [1,1],              # CRC-1 (Parity):   x+1
            None,
            [1,0,1,1],          # CRC-3-GSM:        x3+x+1
            [1,0,0,1,1],        # CRC-4-ITU:        x4+x+1
            [1,1,0,1,0,1]]      # CRC-5-ITU:        x5+x4+x2+1

# **********************************************************************************************************************
# getCrc: Given a bit-stream and a polynomial bit array, this function calculates and returns the CRC.
def getCrc(dataBitArray, polyBitArray):
    polyLen = len(polyBitArray)
    numPad = polyLen-1
    paddedBitArray = np.append(dataBitArray, [0]*(polyLen-1))
    n = len(dataBitArray)
    for d in range(n):
        if paddedBitArray[d]:
            paddedBitArray[d:d+polyLen] ^= polyBitArray
    return paddedBitArray[n:]

# **********************************************************************************************************************
# checkCrc: Given a bit-stream and a polynomial bit array, this function checks the CRC at the end of
# the provided bit-stream and returns the original message without CRC if the check passes. Otherwise,
# this function returns None.
def checkCrc(dataBitArray, polyBitArray):
    polyLen = len(polyBitArray)
    numPad = polyLen-1
    paddedBitArray = np.append(dataBitArray, [0]*(polyLen-1))
    n = len(dataBitArray)
    for d in range(n):
        if paddedBitArray[d]:
            paddedBitArray[d:d+polyLen] ^= polyBitArray
    return None if paddedBitArray[n] else dataBitArray[:n-polyLen+1]

# **********************************************************************************************************************
# appendCrc: Given a bit-stream and a polynomial bit array, this function calculates the CRC, appends
# it to the end of the provided bit-stream, and returns new bit-stream.
def appendCrc(dataBitArray, polyBitArray):
    return np.append(dataBitArray, getCrc(dataBitArray, polyBitArray))

# **********************************************************************************************************************
# QuantumPolar: This class encapsulates the functionality to decode Polar codes using a quantum circuit.
class QuantumPolar:
    def __init__(self, n, k, nCrc=0, **kwargs):
        self.counts = None
        self.topKcounts = None
        self.circuit = None
        self._backend = None
        self.fakeComputer = None
        
        self.device = kwargs.get("device", None)
        if self.device is None and PLATFORM=="BRAKET": self.device = LocalSimulator()
        self.setSimulator(kwargs.get("simulator", "aer_simulator"))

        self.simplify = kwargs.get("simplify", True)
        
        self.n = n
        self.k = k
        self.nCrc = nCrc

        d = int(np.log2(n))
        self.rate = self.k/self.n

        rs = getReliabilitySequence(self.n)
        
        f = n-k-self.nCrc
        self.frozenBits = rs[:f]
        self.msgBits = rs[f:n]    # Total message bits = k+nCrc
        self.g = [1]
        for _ in range(d): self.g = np.kron([[1, 0], [1, 1]], self.g)

    # ******************************************************************************************************************
    @classmethod
    def setPlatform(cls,platform):
        global PLATFORM
        if platform != PLATFORM:
            print("Software platform changed to '%s'."%(platform))
        PLATFORM = platform

    # ******************************************************************************************************************
    @property
    def backend(self):
        if self._backend is None:
            self._backend = Aer.get_backend(self.simulator)
            if self.device is not None: self._backend.set_options(device=self.device)
        return self._backend
        
    # ******************************************************************************************************************
    def setSimulator(self, sim):
        self.simulator = sim
        if sim[:4].lower()=="fake":
            # For more info about "Fake" Quantum Computers visit:
            #   https://qiskit.org/documentation/tutorials/simulators/2_device_noise_simulation.html
            assert PLATFORM=="QISKIT", "Fake Quantum Computers are only supported with QISKIT software platform!"
            fakeComputerClass = getattr(fakes, sim)
            fakeBackend = fakeComputerClass()
            self.fakeComputer = AerSimulator.from_backend(fakeBackend)
            self._backend = fakeBackend
        elif PLATFORM=="QISKIT":
            self._backend = Aer.get_backend(self.simulator)
            if self.device is not None: self._backend.set_options(device=self.device)

    # ******************************************************************************************************************
    def getRandomMessage(self, ebNoDb=4, count=1):
        msgs = np.random.randint(0,2,(count,self.k))
        msgCrcs = msgs if self.nCrc==0 else np.int8([appendCrc(msg,crcPolys[self.nCrc]) for msg in msgs])
        
        u = np.zeros((count,self.n), dtype=np.int8)
        u[:,self.msgBits] = msgCrcs

        codeWords = u.dot(self.g)%2
        tBpsk = 1 - 2.0*codeWords
        
        if type(ebNoDb) in [list, np.ndarray]:  ebNoDb = np.array(ebNoDb)
        else:                                   ebNoDb = np.array([ebNoDb])
        ebNo = 10.0**(ebNoDb/10)
        sigma = 1/np.sqrt(2*self.rate*ebNo)
        sigma2 = sigma*sigma
        
        rx = np.random.normal(tBpsk, np.expand_dims(sigma,(1,2)))
        if len(ebNoDb) == 1: rx,sigma2 = rx[0],sigma2[0]

        if count == 1:
            return msgs[0], msgCrcs[0], u[0], codeWords[0], tBpsk[0], np.squeeze(rx), sigma2
        return msgs, msgCrcs, u, codeWords, tBpsk, rx, sigma2
    
    # ******************************************************************************************************************
    def testSc(self, msg, noise):
        rs = getReliabilitySequence(self.n)
        scFrozenBits = rs[:self.n-self.k]
        scMsgBits = rs[self.n-self.k:self.n]
        
        u = np.int8(self.n*[0])
        u[scMsgBits] = msg
        codeWord = u.dot(self.g)%2
        tBpsk = 1 - 2.0*codeWord
        rx = tBpsk + noise
        uHat, cwHat = scDecode(rx, scFrozenBits)
        msgHat = np.int8(uHat[scMsgBits])
        return msgHat
    
    # ******************************************************************************************************************
    def qbsStr2bsStr(self, qbs):
        return qbs[::-1] if PLATFORM=="QISKIT" else qbs

    # ******************************************************************************************************************
    def bsStr2u(self, bsStr):
        bs = str2bs(bsStr)
        if self.simplify:
            u = np.int8([0 for _ in range(self.n)])
            for i,m in enumerate(sorted(self.msgBits)): u[m] = bs[i]
            return u
            
        return bs
        
    # ******************************************************************************************************************
    def getQuantumPolarGen(self):
        depth = int(np.log2(self.n))
        cNotGates = []
        for d in range(depth):
            for start in range(1<<d):                   # Number of starts:                 1, 2, 4, ..., 2^(d-1)
                for i in range(start,self.n,2<<d):      # Number of repeats for each round: 2^(d-1), ..., 4, 2, 1
                    cNotGates += [ (i+(1<<d), i) ]
        
        if not self.simplify:
            return cNotGates, list(range(self.n)), self.n
        
        keepFrozen = []
        for i,j in cNotGates:
            if (i in self.frozenBits) and (j not in self.frozenBits):
                keepFrozen += [i]

        bitToQubit=[None]*self.n
        numQubits = 0
        for i in range(self.n):
            if (i in self.frozenBits) and (i not in keepFrozen):  continue
            bitToQubit[i] = numQubits
            numQubits += 1

        cNotGates = []
        for d in range(depth):
            for start in range(1<<d):                   # Number of starts:                 1, 2, 4, ..., 2^(d-1)
                for i in range(start,self.n,2<<d):      # Number of repeats for each round: 2^(d-1), ..., 4, 2, 1
                    if bitToQubit[i+(1<<d)] is None: continue
                    if bitToQubit[i] is None:        continue
                    cNotGates += [ (bitToQubit[i+(1<<d)], bitToQubit[i]) ]

        return cNotGates, bitToQubit, numQubits
        
    # ******************************************************************************************************************
    def buildCircuit(self, rx, sigma2):
        cNotGates, bitToQubit, self.numQubits = self.getQuantumPolarGen()
        polarCirc = QuantumCircuit(self.numQubits) if PLATFORM=="QISKIT" else Circuit()

        # Quantum Soft Decision
        for i,q in enumerate(bitToQubit):
            if q is None: continue
            # We can initialize based on the probabilities:
            prob1 = 1/(1+np.exp(2*rx[i]/sigma2))    # Probability of i'th bit being 1
            theta = np.arcsin(np.sqrt(prob1))*2     # Projection onto Z axis gives probability magnitude
            
            # Rotate based on probability => Soft decision
            if PLATFORM=="QISKIT":  polarCirc.ry(theta, q)
            else:                   polarCirc.ry(q, theta)
            
        if PLATFORM=="QISKIT":       polarCirc.barrier()
        
        # Quantum Polar Generator
        for q1, q2 in cNotGates:
            if PLATFORM=="QISKIT":  polarCirc.cx(q1, q2)
            else:                   polarCirc.cnot(control=q1, target=q2)

        if PLATFORM=="QISKIT":      polarCirc.measure_all()
        return polarCirc

    # ******************************************************************************************************************
    def decodeScl(self, rx, listSize):
        uHats, cwHats, pms = sclDecode(rx, self.frozenBits, listSize)
        topCandidate = uHats[0][self.msgBits][:self.k]
        if self.nCrc==0:
            # No CRC -> No "Genie" -> Just return the first candidate
            return topCandidate
            
        for uHat in uHats:
            msgCrc = uHat[self.msgBits]
            msg = checkCrc(msgCrc,crcPolys[self.nCrc])
            if msg is not None:     return msg

        return topCandidate

    # ******************************************************************************************************************
    def run(self, numShots):
        if PLATFORM=="QISKIT":
            if self.fakeComputer is not None:
                tcirc = transpile(self.circuit, self.fakeComputer)
                return self.backend.run(tcirc, seed_simulator=10, shots=numShots).result().get_counts()
                
            return self.backend.run(self.circuit, seed_simulator=10, shots=numShots).result().get_counts()
        
        return self.device.run(self.circuit, shots=numShots).result().measurement_counts

    # ******************************************************************************************************************
    def topKvalidInfo(self, k=None):
        topKCounts = self.topKcounts if k is None else self.topKcounts[:k]

        topKuHats = []
        for bsStr,_ in topKCounts:
            if self.simplify:
                bsMsgBits = str2bs(bsStr[::-1]) if PLATFORM=="QISKIT" else str2bs(bsStr)
                u = np.int8([0 for _ in range(self.n)])
                for i,m in enumerate(sorted(self.msgBits)): u[m] = bsMsgBits[i]
            else:
                u = str2bs(bsStr[::-1]) if PLATFORM=="QISKIT" else str2bs(bsStr)
            topKuHats += [u]
        
        topKuHats = np.int8(topKuHats)
        
        # Remove the ones containing non-zero frozen bits
        validIdx = np.where(topKuHats[:,self.frozenBits].sum(1)==0)[0].tolist()
        
        # If we have CRC, remove the ones with failed CRC check
        if self.nCrc>0:
            newValidIdx = []
            for i in validIdx:
                uHat = topKuHats[i]
                msgCrcHat = uHat[self.msgBits]
                msgHat = checkCrc(msgCrcHat, crcPolys[self.nCrc])
                if msgHat is None: continue    # Skip if CRC check failed
                
                newValidIdx += [i]
            validIdx = newValidIdx
        
        topKuHats = topKuHats[ validIdx ]
        topKmsgHats = topKuHats[:, self.msgBits][:,:self.k]
        topKcwHats = topKuHats.dot(self.g)%2

        return topKuHats, topKmsgHats, topKcwHats

    # ******************************************************************************************************************
    def decode(self, rx, sigma2=0.5, numShots=1024, topK=None, executeCircut=True):
        if executeCircut:
            self.circuit = self.buildCircuit(rx, sigma2)
            self.counts = self.run(numShots)

        self.topKcounts = sorted(self.counts.items(), key=lambda x: -x[1])
        if topK is not None: self.topKcounts = self.topKcounts[:topK]

        topKuHats, topKmsgHats, topKcwHats = self.topKvalidInfo()
        
        if len(topKcwHats)==0:
            return np.int8([0]*self.k) # There is no valid candidates! => Doesn't matter what we return!
        
        # Calculate the correlations with top-K codewords:
        correlations = (rx*(1-2.0*topKcwHats)).sum(axis=1)
        bestIdx = np.argmax(correlations)
        return topKmsgHats[bestIdx]

    # ******************************************************************************************************************
    def decodeFast(self, rx, sigma2=0.5, numShots=1024, topK=None, executeCircut=True):
        if executeCircut:
            self.circuit = self.buildCircuit(rx, sigma2)
            self.counts = self.run(numShots)

        self.topKcounts = sorted(self.counts.items(), key=lambda x: -x[1])
        if topK is not None: self.topKcounts = self.topKcounts[:topK]

        msgHat = None
        for bsStr,_ in self.topKcounts:
            if self.simplify:
                bsMsgBits = str2bs(bsStr[::-1]) if PLATFORM=="QISKIT" else str2bs(bsStr)
                uHat = np.int8([0 for _ in range(self.n)])
                for i,m in enumerate(sorted(self.msgBits)): uHat[m] = bsMsgBits[i]
            else:
                uHat = str2bs(bsStr[::-1]) if PLATFORM=="QISKIT" else str2bs(bsStr)
            
            msgCrcHat = uHat[self.msgBits]
            if msgHat is None: msgHat = msgCrcHat[:self.k]
            if sum(uHat[self.frozenBits])>0:    continue    # Skip if any non-zero Frozen Bit
            
            if self.nCrc==0:                    break       # Found the best

            msg = checkCrc(msgCrcHat, crcPolys[self.nCrc])
            if msg is None:                     continue    # Skip if CRC check failed
            msgHat = msg
            break
        
        return msgHat

    # ******************************************************************************************************************
    def printStats(self, rx, msgDcd, msgGt, k=None, histogram=True):
        topKCounts = self.topKcounts if k is None else self.topKcounts[:k]

        if k is None:   print("\nAll Candidates:")
        else:           print("\nTop-%d Candidates:"%(k))
        
        wB = max(6,self.numQubits)    # Qubits
        wU = self.n+2                 # U
        wM = max(3,self.k)            # MSG
        wC = max(3,self.nCrc)         # CRC
        wCw = max(9,self.n)           # Code-Word

        uStr = "U-" + "".join("M" if i in self.msgBits else "F" for i in range(self.n))
        if self.nCrc>0:
            headerFormat = "    %%-%ds  Count  %%-%ds  %%-%ds  %%-%ds  %%-%ds  Corr."%(wB, wU, wM, wC, wCw)
            print(headerFormat%("Qubits", uStr, "MSG", "CRC", "Code-Word"))
        else:
            headerFormat = "    %%-%ds  Count  %%-%ds  %%-%ds  %%-%ds  Corr."%(wB, wU, wM, wCw)
            print(headerFormat%("Qubits", uStr, "MSG", "Code-Word"))

        for qbsStr,count in topKCounts:
            bsStr = self.qbsStr2bsStr(qbsStr)
            uHat = self.bsStr2u(bsStr)
            uStr = bs2str(uHat)
            cwHat = uHat.dot(self.g)%2
            msgCrcHat = uHat[self.msgBits]
            frozenOk = uHat[self.frozenBits].sum()==0

            if self.nCrc>0:
                if frozenOk:
                    msgHat = checkCrc(msgCrcHat, crcPolys[self.nCrc])
                    msgStr = bs2str(msgHat) if msgHat is not None else "N/A"
                    crcStr = bs2str(msgCrcHat[self.k:])
                    cwStr = bs2str(cwHat) if msgHat is not None else "N/A"
                else:
                    msgStr = cwStr = crcStr = "N/A"
                
                correlation = (rx*(1-2.0*cwHat)).sum()
                
                if not frozenOk:                flag = "F"
                elif msgHat is None:            flag = "C"
                elif np.all(msgHat==msgDcd):
                    if np.all(msgHat==msgGt):   flag = '✓'
                    else:                       flag = "D"
                elif np.all(msgHat==msgGt):     flag = "G"
                else:                           flag = ""
                formatStr = "    %%-%ds  %%-%dd  %%-%ds  %%-%ds  %%-%ds  %%-%ds  %%-7.3f %%s"%(wB, 5, wU, wM, wC, wCw)
                print(formatStr%(bsStr, count, "  "+uStr, msgStr, crcStr, cwStr, correlation, flag))
            else:
        
                msgStr = bs2str(msgCrcHat) if frozenOk else "N/A"
                cwStr = bs2str(cwHat) if frozenOk else "N/A"
                correlation = (rx*(1-2.0*cwHat)).sum()
                
                if not frozenOk:                flag = "F"
                elif np.all(msgCrcHat==msgDcd):
                    if np.all(msgDcd==msgGt):   flag = '✓'
                    else:                       flag = "D"
                elif np.all(msgCrcHat==msgGt):  flag = "G"
                else:                           flag = ""
                formatStr = "    %%-%ds  %%-%dd  %%-%ds  %%-%ds  %%-%ds  %%-7.3f %%s"%(wB, 5, wU, wM, wCw)
                print(formatStr%(bsStr, count, "  "+uStr, msgStr, cwStr, correlation, flag))

        if histogram:
            # Finally show a histogram of the bitstreams as measured at the output of the circuit
            bsStrs,counts = zip(*topKCounts)
            plt.bar([self.qbsStr2bsStr(bsStr) for bsStr in bsStrs], list(counts))
            plt.xticks(rotation=90)
            plt.xlabel('bit-stream')
            plt.ylabel('counts')
            plt.show()
