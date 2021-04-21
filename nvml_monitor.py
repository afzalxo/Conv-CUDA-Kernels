from pynvml import *
import time
import threading
import subprocess

class NVMLMonThread(threading.Thread):
    def __init__(self, fname, kwargs=None):
        threading.Thread.__init__(self, kwargs=None)
        nvmlInit()
        self.fname = fname
        self.fhandle = open(self.fname, 'w')
        self.finish = False
        self.handle = nvmlDeviceGetHandleByIndex(0)
        print('Thread Initialized...')

    def run(self):
        handle = self.handle
        trace = []
        cur_trace = ''
        while not self.finish:
            pwr = nvmlDeviceGetPowerUsage(handle)/1000.
            #temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
            meminfo = nvmlDeviceGetMemoryInfo(handle)
            #pcie_util = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_COUNT) #PCIe Throughput in KB/s
            enr = nvmlDeviceGetTotalEnergyConsumption(handle) #Energy consumption in mJ since driver was last loaded
            util = nvmlDeviceGetUtilizationRates(handle) #gpu: Percent of time over the past sample period during which one or more kernels was executing on the GPU.
                                                        #memory: Percent of time over the past sample period during which global (device) memory was being read or written.
            #throttle_reasons = nvmlDeviceGetCurrentClocksThrottleReasons(handle) #Get throttle Reasons
            #print(pwr, temp, meminfo.total/1024./1024./1024., meminfo.free/1024./1024./1024., meminfo.used/1024./1024./1024., pcie_util/1024./1024., enr/1000./1000., util.gpu, util.memory, throttle_reasons)
            cur_trace = str(pwr) + ', ' + str(meminfo.total/1024./1024./1024.) + ', ' + str(meminfo.free/1024./1024./1024.) + ', ' + str(meminfo.used/1024./1024./1024.) + ', ' + str(enr/1000./1000.) + ', ' + str(util.gpu) + ', ' + str(util.memory)+'\n'# + ', ' + str(throttle_reasons) + '\n'
            trace.append(cur_trace)
            #self.fhandle.write(str(pwr) +', ' + str(temp) + ', ' + str(meminfo.total/1024./1024./1024.) + ', ' + str(meminfo.free/1024./1024./1024.) + ', ' + str(meminfo.used/1024./1024./1024.) + ', ' + str(pcie_util/1024./1024.) + ', ' + str(enr/1000./1000.) + ', ' + str(util.gpu) + ', ' + str(util.memory) + ', ' + str(throttle_reasons) + '\n')
            time.sleep(0.015)
        str_trace = ' '.join([str(elem) for elem in trace])
        self.fhandle.write(str_trace)
        self.cleanup()
    
    def cleanup(self):
        self.fhandle.close()
        nvmlShutdown()


monT = NVMLMonThread('trace_gemm_14')
print('Starting Monitoring Thread...')
monT.start()
kern_proc = subprocess.Popen(['./gemm_kernel'])
kern_proc.wait()
monT.finish = True
monT.join()
print('Done. Adios Amigo...')


'''
nvmlInit()
fname = 'trace_0.csv'
devCnt = nvmlDeviceGetCount()
pwr = 0
handle = nvmlDeviceGetHandleByIndex(0)
pwr_lim = nvmlDeviceGetEnforcedPowerLimit(handle)
print('Device Power Limit: '+str(pwr_lim))
clk_cnt = nvmlDeviceGetMaxClockInfo(handle,NVML_CLOCK_MEM)
print('Max Mem Clock: ' + str(clk_cnt))
fhandle = open(fname, 'w')
while(True):
    pwr = nvmlDeviceGetPowerUsage(handle)/1000.
    temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
    meminfo = nvmlDeviceGetMemoryInfo(handle)
    pcie_util = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_COUNT) #PCIe Throughput in KB/s
    enr = nvmlDeviceGetTotalEnergyConsumption(handle) #Energy consumption in mJ since driver was last loaded
    util = nvmlDeviceGetUtilizationRates(handle) #gpu: Percent of time over the past sample period during which one or more kernels was executing on the GPU.
                                                #memory: Percent of time over the past sample period during which global (device) memory was being read or written.
    throttle_reasons = nvmlDeviceGetCurrentClocksThrottleReasons(handle) #Get throttle Reasons
    #print(pwr, temp, meminfo.total/1024./1024./1024., meminfo.free/1024./1024./1024., meminfo.used/1024./1024./1024., pcie_util/1024./1024., enr/1000./1000., util.gpu, util.memory, throttle_reasons)
    fhandle.write(str(pwr) +', ' + str(temp) + ', ' + str(meminfo.total/1024./1024./1024.) + ', ' + str(meminfo.free/1024./1024./1024.) + ', ' + str(meminfo.used/1024./1024./1024.) + ', ' + str(pcie_util/1024./1024.) + ', ' + str(enr/1000./1000.) + ', ' + str(util.gpu) + ', ' + str(util.memory) + ', ' + str(throttle_reasons) + '\n')
    time.sleep(0.01)
'''
