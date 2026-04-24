#include <cstddef>
#include <stdexcept>
#include <string>

#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

void* wilcoxon_rmm_allocate(size_t bytes) {
    try {
        return rmm::mr::get_current_device_resource()->allocate_sync(bytes);
    } catch (std::exception const& e) {
        throw std::runtime_error(
            std::string("RMM allocation failed in Wilcoxon scratch: ") +
            e.what());
    }
}

void wilcoxon_rmm_deallocate(void* ptr, size_t bytes) {
    rmm::mr::get_current_device_resource()->deallocate_sync(ptr, bytes);
}
