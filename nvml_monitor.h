#ifndef NVMLMONTHREAD_H_
#define NVMLMONTHREAD_H_

#include <chrono>
#include <nvml.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

class NVMLMonThread {
	public:
		nvmlDevice_t devhandle;
		std::ofstream fhandle;
		typedef struct _datum {
			std::time_t timestamp;
			uint temperature;
			uint power;
			nvmlUtilization_t utilization;
			uint state;
		} datum;
		std::vector <datum> log_enum;
		bool loop;
		uint caller_state;


		NVMLMonThread(int devID, std::string const &fname) {
			nvmlInit();
			nvmlDeviceGetHandleByIndex(devID, &devhandle);
			log_enum.reserve(200000);
			fhandle.open(fname, std::ios::out);
			caller_state = 0;
			printf("%li, %li\n", std::chrono::high_resolution_clock::duration::period::num, std::chrono::high_resolution_clock::duration::period::den);
		}

		~NVMLMonThread() {
			nvmlShutdown();
			store_log();
		}

		void log() {
			datum point {};
			loop = true;
			while(loop){
				point.timestamp = std::chrono::high_resolution_clock::now( ).time_since_epoch( ).count( );
				nvmlDeviceGetTemperature( devhandle, NVML_TEMPERATURE_GPU, &point.temperature);
				nvmlDeviceGetPowerUsage(devhandle, &point.power);
				nvmlDeviceGetUtilizationRates(devhandle, &point.utilization);
				point.state = caller_state;
				log_enum.push_back(point);
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
		}

		void killThread() {
			std::this_thread::sleep_for(std::chrono::seconds(2));
			loop = false;
		}

		void store_log() {
			for(int i = 0; i < static_cast<int>(log_enum.size()); i++){
				fhandle << log_enum[i].timestamp << ", " << log_enum[i].temperature << ", " << log_enum[i].power << ", " << log_enum[i].utilization.gpu << ", " << log_enum[i].utilization.memory << ", " << log_enum[i].state << "\n";
			}
			fhandle.close();
		}
};

#endif
