using Silk.NET.Vulkan;
using System;
using System.Collections.Generic;
using System.Threading;

namespace Ryujinx.Graphics.Vulkan
{
    /// <summary>
    /// Recycles fence handles, as creating and destroying fences can be expensive on some drivers.
    /// </summary>
    class FencePool : IDisposable
    {
        private const int MaxPooledFences = 64;

        private readonly Vk _api;
        private readonly Device _device;
        private readonly Stack<Fence> _fences = new();
        private readonly Lock _lock = new();
        private bool _disposed;

        public FencePool(Vk api, Device device)
        {
            _api = api;
            _device = device;
        }

        /// <summary>
        /// Gets an unsignaled fence, reusing a previously returned one if possible.
        /// </summary>
        /// <returns>The fence</returns>
        public unsafe Fence Rent()
        {
            Fence fence;

            lock (_lock)
            {
                _fences.TryPop(out fence);
            }

            if (fence.Handle != 0)
            {
                _api.ResetFences(_device, 1, in fence).ThrowOnError();

                return fence;
            }

            FenceCreateInfo fenceCreateInfo = new()
            {
                SType = StructureType.FenceCreateInfo,
            };

            _api.CreateFence(_device, in fenceCreateInfo, null, out fence).ThrowOnError();

            return fence;
        }

        /// <summary>
        /// Returns a fence to the pool. The fence must not be used by any pending queue submission.
        /// </summary>
        /// <param name="fence">The fence to return</param>
        public void Return(Fence fence)
        {
            lock (_lock)
            {
                if (!_disposed && _fences.Count < MaxPooledFences)
                {
                    _fences.Push(fence);

                    return;
                }
            }

            _api.DestroyFence(_device, fence, Span<AllocationCallbacks>.Empty);
        }

        public void Dispose()
        {
            lock (_lock)
            {
                _disposed = true;

                while (_fences.TryPop(out Fence fence))
                {
                    _api.DestroyFence(_device, fence, Span<AllocationCallbacks>.Empty);
                }
            }
        }
    }
}
