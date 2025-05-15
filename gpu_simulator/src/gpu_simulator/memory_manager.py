"""
Memory Manager - Advanced GPU memory management simulation
"""
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict

logger = logging.getLogger('gpu_simulator.memory_manager')

class Block:
    """Represents a block of memory in the simulated GPU"""
    
    def __init__(self, address: int, size: int, name: str = "", tensor_id: Optional[int] = None):
        self.address = address
        self.size = size
        self.name = name
        self.tensor_id = tensor_id
        self.allocated_at = time.time()
        self.last_used = self.allocated_at
    
    def __str__(self) -> str:
        return f"Block(addr={self.address}, size={self.size}, name='{self.name}')"


class MemoryManager:
    """Advanced memory manager for simulated GPU memory"""
    
    def __init__(self, total_memory: int, device_id: int = 0):
        """
        Initialize the memory manager
        
        Args:
            total_memory: Total memory in MB
            device_id: GPU device ID
        """
        self.total_memory = total_memory
        self.device_id = device_id
        
        # Track allocated and free blocks
        self.allocated_blocks: Dict[int, Block] = {}  # address -> Block
        self.free_blocks: List[Tuple[int, int]] = [(0, total_memory)]  # (start_addr, size)
        
        # Statistics
        self.peak_memory = 0
        self.num_allocations = 0
        self.num_deallocations = 0
        self.allocation_sizes: List[int] = []
        self.fragmentation_history: List[float] = []
        
        # Cache management
        self.enable_caching = True
        self.cache: Dict[str, int] = {}  # tensor_hash -> address
        
        logger.info(f"Initialized MemoryManager for GPU {device_id} with {total_memory}MB")
    
    def allocate(self, size: int, name: str = "", tensor_id: Optional[int] = None) -> int:
        """
        Allocate a block of memory
        
        Args:
            size: Size in MB
            name: Name for the allocation
            tensor_id: ID of the tensor (for cache management)
            
        Returns:
            address: The starting address of the allocated block
        """
        # Check if in cache
        tensor_hash = f"{name}_{size}"
        if self.enable_caching and tensor_hash in self.cache:
            cached_addr = self.cache[tensor_hash]
            if cached_addr in self.allocated_blocks:
                block = self.allocated_blocks[cached_addr]
                block.last_used = time.time()
                logger.debug(f"Cache hit for '{name}' ({size}MB) at address {cached_addr}")
                return cached_addr
        
        # Find a suitable free block with first-fit strategy
        block_index = -1
        for i, (start_addr, block_size) in enumerate(self.free_blocks):
            if block_size >= size:
                block_index = i
                break
        
        if block_index == -1:
            # No suitable free block found, try to free up memory
            if not self._try_compact():
                # If compaction doesn't help, try to evict least recently used tensors
                if not self._try_evict(size):
                    raise MemoryError(f"Failed to allocate {size}MB for '{name}': Out of memory and "
                                      f"could not free enough space even after compaction and eviction")
            
            # Try again after memory management
            return self.allocate(size, name, tensor_id)
        
        # Allocate from the found free block
        start_addr, block_size = self.free_blocks.pop(block_index)
        
        # Create the new allocated block
        new_block = Block(start_addr, size, name, tensor_id)
        self.allocated_blocks[start_addr] = new_block
        
        # Update statistics
        self.num_allocations += 1
        self.allocation_sizes.append(size)
        currently_allocated = sum(block.size for block in self.allocated_blocks.values())
        self.peak_memory = max(self.peak_memory, currently_allocated)
        
        # If there's remaining space in the block, add it back to free blocks
        if block_size > size:
            self.free_blocks.append((start_addr + size, block_size - size))
            # Sort free blocks by address for better compaction later
            self.free_blocks.sort()
        
        # Update cache
        if self.enable_caching and tensor_id is not None:
            self.cache[tensor_hash] = start_addr
        
        # Calculate fragmentation
        self._update_fragmentation()
        
        logger.debug(f"Allocated {size}MB for '{name}' at address {start_addr}")
        return start_addr
    
    def free(self, address: int) -> None:
        """
        Free a previously allocated block
        
        Args:
            address: The address of the block to free
        """
        if address not in self.allocated_blocks:
            raise ValueError(f"Invalid address to free: {address}")
        
        block = self.allocated_blocks.pop(address)
        
        # Add to free blocks
        self.free_blocks.append((address, block.size))
        
        # Sort free blocks by address
        self.free_blocks.sort()
        
        # Merge adjacent free blocks
        self._merge_free_blocks()
        
        # Update statistics
        self.num_deallocations += 1
        
        # Update fragmentation
        self._update_fragmentation()
        
        logger.debug(f"Freed {block.size}MB from '{block.name}' at address {address}")
    
    def _merge_free_blocks(self) -> None:
        """Merge adjacent free blocks to reduce fragmentation"""
        if not self.free_blocks:
            return
        
        merged_blocks = []
        current_block = self.free_blocks[0]
        
        for i in range(1, len(self.free_blocks)):
            next_block = self.free_blocks[i]
            curr_addr, curr_size = current_block
            next_addr, next_size = next_block
            
            # Check if blocks are adjacent
            if curr_addr + curr_size == next_addr:
                # Merge blocks
                current_block = (curr_addr, curr_size + next_size)
            else:
                merged_blocks.append(current_block)
                current_block = next_block
        
        merged_blocks.append(current_block)
        self.free_blocks = merged_blocks
    
    def _try_compact(self) -> bool:
        """Try to compact memory to reduce fragmentation"""
        # This is a simplified compaction simulation
        # In a real GPU, compaction is much more complex
        
        # If we have multiple free blocks, it means we have fragmentation
        if len(self.free_blocks) <= 1:
            return False
        
        # Merge free blocks
        old_free_size = sum(size for _, size in self.free_blocks)
        self._merge_free_blocks()
        new_free_size = sum(size for _, size in self.free_blocks)
        
        # Check if compaction helped
        compaction_gain = len(self.free_blocks) > 1 and new_free_size > old_free_size
        
        if compaction_gain:
            logger.info(f"Memory compaction: reduced from {len(self.free_blocks)} to "
                       f"{len(self.free_blocks)} blocks, freed additional {new_free_size - old_free_size}MB")
        
        return compaction_gain
    
    def _try_evict(self, required_size: int) -> bool:
        """
        Try to evict least recently used tensors to free up memory
        
        Args:
            required_size: The amount of memory needed in MB
            
        Returns:
            bool: True if enough memory was freed, False otherwise
        """
        if not self.allocated_blocks:
            return False
        
        # Sort blocks by last used time
        blocks_by_lru = sorted(
            self.allocated_blocks.items(),
            key=lambda x: x[1].last_used
        )
        
        # Calculate total free memory
        free_memory = sum(size for _, size in self.free_blocks)
        
        # Evict blocks until we have enough memory
        evicted = []
        for addr, block in blocks_by_lru:
            # Skip blocks that are marked as non-evictable
            if block.name.startswith("_pinned_"):
                continue
                
            evicted.append((addr, block))
            free_memory += block.size
            
            if free_memory >= required_size:
                break
        
        # If we still don't have enough memory, eviction failed
        if free_memory < required_size:
            return False
        
        # Actually evict the blocks
        for addr, block in evicted:
            logger.info(f"Evicting '{block.name}' ({block.size}MB) to free memory")
            self.free(addr)
        
        return True
    
    def _update_fragmentation(self) -> None:
        """Calculate and update memory fragmentation statistics"""
        if not self.free_blocks:
            self.fragmentation_history.append(0.0)
            return
        
        # Total free memory
        total_free = sum(size for _, size in self.free_blocks)
        
        # Largest contiguous block
        largest_block = max(size for _, size in self.free_blocks)
        
        # Fragmentation is measured as 1 - (largest_block / total_free)
        # If all free memory is in one block, fragmentation is 0
        # If free memory is scattered, fragmentation approaches 1
        if total_free > 0:
            fragmentation = 1.0 - (largest_block / total_free)
        else:
            fragmentation = 0.0
        
        self.fragmentation_history.append(fragmentation)
        
        if fragmentation > 0.5:
            logger.warning(f"High memory fragmentation detected: {fragmentation:.2f}")
    
    def memory_info(self) -> Dict:
        """Get detailed memory information"""
        allocated = sum(block.size for block in self.allocated_blocks.values())
        free = self.total_memory - allocated
        
        # Count allocation sizes by category
        size_categories = {
            "small (< 1MB)": 0,
            "medium (1-10MB)": 0,
            "large (10-100MB)": 0,
            "xlarge (> 100MB)": 0
        }
        
        for block in self.allocated_blocks.values():
            if block.size < 1:
                size_categories["small (< 1MB)"] += 1
            elif block.size < 10:
                size_categories["medium (1-10MB)"] += 1
            elif block.size < 100:
                size_categories["large (10-100MB)"] += 1
            else:
                size_categories["xlarge (> 100MB)"] += 1
        
        # Get current fragmentation
        current_fragmentation = self.fragmentation_history[-1] if self.fragmentation_history else 0
        
        return {
            "total_memory": self.total_memory,
            "allocated_memory": allocated,
            "free_memory": free,
            "utilization": (allocated / self.total_memory) * 100,
            "num_allocated_blocks": len(self.allocated_blocks),
            "num_free_blocks": len(self.free_blocks),
            "largest_free_block": max((size for _, size in self.free_blocks), default=0),
            "fragmentation": current_fragmentation,
            "allocation_by_size": size_categories,
            "peak_memory_usage": self.peak_memory,
            "num_allocations": self.num_allocations,
            "num_deallocations": self.num_deallocations
        }
    
    def reset_stats(self) -> None:
        """Reset memory statistics"""
        self.peak_memory = sum(block.size for block in self.allocated_blocks.values())
        self.num_allocations = len(self.allocated_blocks)
        self.num_deallocations = 0
        self.allocation_sizes = []
        self.fragmentation_history = []
    
    def enable_cache(self, enabled: bool = True) -> None:
        """Enable or disable memory caching"""
        self.enable_caching = enabled
        if not enabled:
            self.cache.clear()
    
    def pin_memory(self, address: int) -> None:
        """Pin a memory block to prevent eviction"""
        if address in self.allocated_blocks:
            block = self.allocated_blocks[address]
            if not block.name.startswith("_pinned_"):
                block.name = f"_pinned_{block.name}"
    
    def unpin_memory(self, address: int) -> None:
        """Unpin a memory block"""
        if address in self.allocated_blocks:
            block = self.allocated_blocks[address]
            if block.name.startswith("_pinned_"):
                block.name = block.name[8:]  # Remove "_pinned_" prefix
    
    def __str__(self) -> str:
        info = self.memory_info()
        return (f"MemoryManager(GPU={self.device_id}, "
                f"memory={info['allocated_memory']}/{self.total_memory}MB, "
                f"frag={info['fragmentation']:.2f})")