using Silk.NET.Vulkan;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading;

namespace Ryujinx.Graphics.Vulkan
{
    class MemoryAllocator : IDisposable
    {
        private const ulong MaxDeviceMemoryUsageEstimate = 16UL * 1024 * 1024 * 1024;

        // Preferred block sizes, matching the Vulkan Memory Allocator defaults.
        private const ulong LargeHeapBlockSize = 256UL * 1024 * 1024;
        private const ulong SmallHeapMaxSize = 1UL * 1024 * 1024 * 1024;
        private const ulong MaxBlockSize = 1UL << 30;

        private readonly Vk _api;
        private readonly VulkanPhysicalDevice _physicalDevice;
        private readonly Device _device;
        private readonly List<MemoryAllocatorBlockList> _blockLists;
        private readonly int _blockAlignment;
        private readonly ReaderWriterLockSlim _lock;

        public MemoryAllocator(Vk api, VulkanPhysicalDevice physicalDevice, Device device)
        {
            _api = api;
            _physicalDevice = physicalDevice;
            _device = device;
            _blockLists = [];
            _blockAlignment = (int)Math.Min(int.MaxValue, MaxDeviceMemoryUsageEstimate / _physicalDevice.PhysicalDeviceProperties.Limits.MaxMemoryAllocationCount);
            _lock = new(LockRecursionPolicy.NoRecursion);
        }

        public MemoryAllocation AllocateDeviceMemory(
            MemoryRequirements requirements,
            MemoryPropertyFlags flags = 0,
            bool isBuffer = false)
        {
            int memoryTypeIndex = FindSuitableMemoryTypeIndex(requirements.MemoryTypeBits, flags);
            if (memoryTypeIndex < 0)
            {
                return default;
            }

            bool map = flags.HasFlag(MemoryPropertyFlags.HostVisibleBit);
            return Allocate(memoryTypeIndex, requirements.Size, requirements.Alignment, map, isBuffer);
        }

        private MemoryAllocation Allocate(int memoryTypeIndex, ulong size, ulong alignment, bool map, bool isBuffer)
        {
            _lock.EnterReadLock();

            try
            {
                for (int i = 0; i < _blockLists.Count; i++)
                {
                    MemoryAllocatorBlockList bl = _blockLists[i];
                    if (bl.MemoryTypeIndex == memoryTypeIndex && bl.ForBuffer == isBuffer)
                    {
                        return bl.Allocate(size, alignment, map);
                    }
                }
            }
            finally
            {
                _lock.ExitReadLock();
            }

            _lock.EnterWriteLock();

            try
            {
                MemoryAllocatorBlockList newBl = new(_api, _device, memoryTypeIndex, GetBlockSize(memoryTypeIndex), isBuffer);
                _blockLists.Add(newBl);

                return newBl.Allocate(size, alignment, map);
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        private int GetBlockSize(int memoryTypeIndex)
        {
            // Some drivers report an effectively unlimited allocation count, which would make every
            // allocation its own block. Memory allocations are expensive, so prefer large blocks.
            uint heapIndex = _physicalDevice.PhysicalDeviceMemoryProperties.MemoryTypes[memoryTypeIndex].HeapIndex;
            ulong heapSize = _physicalDevice.PhysicalDeviceMemoryProperties.MemoryHeaps[(int)heapIndex].Size;

            ulong preferredSize = heapSize <= SmallHeapMaxSize ? heapSize / 8 : LargeHeapBlockSize;
            ulong blockSize = Math.Clamp(Math.Max((ulong)_blockAlignment, preferredSize), 1UL, MaxBlockSize);

            // Block sizes are used as an alignment, so they must be a power of two.
            return (int)(1UL << (63 - BitOperations.LeadingZeroCount(blockSize)));
        }

        internal int FindSuitableMemoryTypeIndex(uint memoryTypeBits, MemoryPropertyFlags flags)
        {
            for (int i = 0; i < _physicalDevice.PhysicalDeviceMemoryProperties.MemoryTypeCount; i++)
            {
                MemoryType type = _physicalDevice.PhysicalDeviceMemoryProperties.MemoryTypes[i];

                if ((memoryTypeBits & (1 << i)) != 0)
                {
                    if (type.PropertyFlags.HasFlag(flags))
                    {
                        return i;
                    }
                }
            }

            return -1;
        }

        public static bool IsDeviceMemoryShared(VulkanPhysicalDevice physicalDevice)
        {
            for (int i = 0; i < physicalDevice.PhysicalDeviceMemoryProperties.MemoryHeapCount; i++)
            {
                if (!physicalDevice.PhysicalDeviceMemoryProperties.MemoryHeaps[i].Flags.HasFlag(MemoryHeapFlags.DeviceLocalBit))
                {
                    return false;
                }
            }

            return true;
        }

        public void Dispose()
        {
            for (int i = 0; i < _blockLists.Count; i++)
            {
                _blockLists[i].Dispose();
            }
        }
    }
}
