NVIDIA_TESLA_V100 = "V100"
NVIDIA_TESLA_P100 = "P100"
NVIDIA_TESLA_T4 = "T4"
NVIDIA_TESLA_P4 = "P4"
NVIDIA_TESLA_K80 = "K80"
NVIDIA_TESLA_A10G = "A10G"
NVIDIA_L4 = "L4"
NVIDIA_A100 = "A100"
INTEL_MAX_1550 = "Intel-GPU-Max-1550"
INTEL_MAX_1100 = "Intel-GPU-Max-1100"
INTEL_GAUDI = "Intel-GAUDI"
AMD_INSTINCT_MI100 = "AMD-Instinct-MI100"
AMD_INSTINCT_MI250x = "AMD-Instinct-MI250X"
AMD_INSTINCT_MI250 = "AMD-Instinct-MI250X-MI250"
AMD_INSTINCT_MI210 = "AMD-Instinct-MI210"
AMD_INSTINCT_MI300x = "AMD-Instinct-MI300X-OAM"
AMD_RADEON_R9_200_HD_7900 = "AMD-Radeon-R9-200-HD-7900"
AMD_RADEON_HD_7900 = "AMD-Radeon-HD-7900"
AWS_NEURON_CORE = "aws-neuron-core"
GOOGLE_TPU_V2 = "TPU-V2"
GOOGLE_TPU_V3 = "TPU-V3"
GOOGLE_TPU_V4 = "TPU-V4"
GOOGLE_TPU_V5P = "TPU-V5P"
GOOGLE_TPU_V5LITEPOD = "TPU-V5LITEPOD"

# Use these instead of NVIDIA_A100 if you need a specific accelerator size. Note that
# these labels are not auto-added to nodes, you'll have to add them manually in
# addition to the default A100 label if needed.
NVIDIA_A100_40G = "A100-40G"
NVIDIA_A100_80G = "A100-80G"
