//
//  HardwareCompat.swift
//  LidAngleSensor
//
//  Created by Sam on 2026-03-22.
//

import Foundation
import IOKit.hid


/// Result of runtime HID probing.
enum LASSensorProbeResult: CustomStringConvertible {
    /// Found sensor on the standard Sensor usage page (0x0020).
    case foundStandard(device: IOHIDDevice)
    /// Found 0x8104 device on vendor-specific usage page (0xFF00).
    /// Hardware exists but we can't read angle data through this interface.
    case foundVendorSpecific
    /// No matching HID device at all.
    case notFound
    
    var description: String {
        switch self {
        case .foundStandard: "found (standard UsagePage 0x0020)"
        case .foundVendorSpecific: "found (vendor-specific UsagePage 0xFF00)"
        case .notFound: "not found"
        }
    }
}

// MARK: - Model Identification

struct MacModelInfo {
    
    /// Raw model identifier, e.g. "MacBookPro18,3" or "Mac16,1".
    let identifier: String
    
    /// Read the current Mac's model identifier via sysctl.
    static func current() -> MacModelInfo {
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [UInt8](repeating: 0, count: size)
        sysctlbyname("hw.model", &model, &size, nil, 0)
        let identifier = String(decoding: model.prefix(while: { $0 != 0 }), as: UTF8.self)
        return MacModelInfo(identifier: identifier)
    }
    
}

// MARK: - Runtime HID Probing

extension MacModelInfo {
    
    private static let noOptions = IOOptionBits(kIOHIDOptionsTypeNone)
    
    /// Probe HID subsystem for the lid angle sensor.
    static func probeSensor() -> LASSensorProbeResult {
        // Strategy 1: Standard Sensor page (0x0020) + Orientation (0x008A)
        if let device = findHIDDevice(usagePage: 0x0020, usage: 0x008A) {
            return .foundStandard(device: device)
        }
        
        // Strategy 2: Check if 0x8104 exists under any usage page
        if deviceExistsWithProductID(0x8104) {
            return .foundVendorSpecific
        }
        
        return .notFound
    }
    
    private static func findHIDDevice(usagePage: Int, usage: Int) -> IOHIDDevice? {
        let manager = IOHIDManagerCreate(kCFAllocatorDefault, noOptions)
        guard IOHIDManagerOpen(manager, noOptions) == kIOReturnSuccess else { return nil }
        defer { IOHIDManagerClose(manager, noOptions) }
        
        let matching: [String: Any] = [
            kIOHIDVendorIDKey as String: 0x05AC,
            kIOHIDProductIDKey as String: 0x8104,
            "UsagePage": usagePage,
            "Usage": usage,
        ]
        
        IOHIDManagerSetDeviceMatching(manager, matching as CFDictionary)
        
        guard let devices = IOHIDManagerCopyDevices(manager) as? Set<IOHIDDevice>,
              !devices.isEmpty
        else { return nil }
        
        for device in devices {
            guard IOHIDDeviceOpen(device, noOptions) == kIOReturnSuccess else { continue }
            defer { IOHIDDeviceClose(device, noOptions) }
            
            var report = [UInt8](repeating: 0, count: 8)
            var length = CFIndex(report.count)
            
            let result = IOHIDDeviceGetReport(
                device,
                kIOHIDReportTypeFeature,
                1,
                &report,
                &length
            )
            
            if result == kIOReturnSuccess, length >= 3 {
                return device
            }
        }
        
        return nil
    }
    
    private static func deviceExistsWithProductID(_ productID: Int) -> Bool {
        let manager = IOHIDManagerCreate(kCFAllocatorDefault, noOptions)
        guard IOHIDManagerOpen(manager, noOptions) == kIOReturnSuccess else { return false }
        defer { IOHIDManagerClose(manager, noOptions) }
        
        let matching: [String: Any] = [
            kIOHIDVendorIDKey as String: 0x05AC,
            kIOHIDProductIDKey as String: productID,
        ]
        
        IOHIDManagerSetDeviceMatching(manager, matching as CFDictionary)
        
        guard let devices = IOHIDManagerCopyDevices(manager) as? Set<IOHIDDevice> else {
            return false
        }
        return !devices.isEmpty
    }
}

// MARK: - Diagnostic Report

struct LASDiagnostic {
    let modelInfo: MacModelInfo
    let probeResult: LASSensorProbeResult
    
    var isSensorUsable: Bool {
        if case .foundStandard = probeResult { return true }
        return false
    }
    
    /// Cached at first access; subsequent calls are free.
    static let shared: LASDiagnostic = {
        let model = MacModelInfo.current()
        let probe = MacModelInfo.probeSensor()
        
        let diag = LASDiagnostic(modelInfo: model, probeResult: probe)
        
        return diag
    }()
    
    static func run() -> LASDiagnostic { shared }
}