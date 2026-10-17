<#
Read-only Core Audio diagnostic. No audio stream, capture, playback, volume change,
registry access, permission change, or dependency installation.

The volume slider is not a signal-energy measurement. Hardware controls are read
only when the capture topology exposes them; missing controls mean unknown.
Listen-to-this-device and vendor APO settings are deliberately not inferred.

Sources: Microsoft Learn IAudioEndpointVolume / IAudioVolumeLevel / DeviceTopology,
and the Microsoft Win32 metadata mmdeviceapi.h and devicetopology.h declarations.
#>
[CmdletBinding()]
param([switch]$CompileOnly, [switch]$SkipTopology)

$ErrorActionPreference = 'Stop'
if (-not [Environment]::Is64BitProcess) { throw 'Run this diagnostic in 64-bit PowerShell.' }

if (-not ('FanyiJiEndpointReadOnly.Probe' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace FanyiJiEndpointReadOnly {
    [StructLayout(LayoutKind.Sequential)]
    public struct PropertyKey {
        public Guid FormatId; public uint Id;
        public PropertyKey(string guid, uint id) { FormatId = new Guid(guid); Id = id; }
    }
    [StructLayout(LayoutKind.Explicit, Size = 24)]
    public struct PropVariant {
        [FieldOffset(0)] public ushort Type;
        [FieldOffset(8)] public IntPtr Pointer;
        [FieldOffset(8)] public uint UIntValue;
    }
    [ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IDeviceEnumerator {
        [PreserveSig] int EnumAudioEndpoints(int flow, uint states, out IDeviceCollection devices);
        [PreserveSig] int GetDefaultAudioEndpoint(int flow, int role, out IDevice device);
    }
    [ComImport, Guid("0BD7A1BE-7A1A-44DB-8397-CC5392387B5E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IDeviceCollection {
        [PreserveSig] int GetCount(out uint count);
        [PreserveSig] int Item(uint index, out IDevice device);
    }
    [ComImport, Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IDevice {
        [PreserveSig] int Activate(ref Guid iid, uint context, IntPtr parameters, [MarshalAs(UnmanagedType.IUnknown)] out object result);
        [PreserveSig] int OpenPropertyStore(uint access, out IPropertyStore store);
        [PreserveSig] int GetId([MarshalAs(UnmanagedType.LPWStr)] out string id);
        [PreserveSig] int GetState(out uint state);
    }
    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IPropertyStore {
        [PreserveSig] int GetCount(out uint count);
        [PreserveSig] int GetAt(uint index, out PropertyKey key);
        [PreserveSig] int GetValue(ref PropertyKey key, out PropVariant value);
    }
    [ComImport, Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IEndpointVolume {
        // Reserved vtable entries are declared but never called by this probe.
        [PreserveSig] int RegisterControlChangeNotify(IntPtr notify);
        [PreserveSig] int UnregisterControlChangeNotify(IntPtr notify);
        [PreserveSig] int GetChannelCount(out uint count);
        [PreserveSig] int ReservedSetMasterDb(float level, IntPtr context);
        [PreserveSig] int ReservedSetMasterScalar(float level, IntPtr context);
        [PreserveSig] int GetMasterVolumeLevel(out float level);
        [PreserveSig] int GetMasterVolumeLevelScalar(out float level);
        [PreserveSig] int ReservedSetChannelDb(uint channel, float level, IntPtr context);
        [PreserveSig] int ReservedSetChannelScalar(uint channel, float level, IntPtr context);
        [PreserveSig] int GetChannelVolumeLevel(uint channel, out float level);
        [PreserveSig] int GetChannelVolumeLevelScalar(uint channel, out float level);
        [PreserveSig] int ReservedSetMute([MarshalAs(UnmanagedType.Bool)] bool mute, IntPtr context);
        [PreserveSig] int GetMute([MarshalAs(UnmanagedType.Bool)] out bool mute);
        [PreserveSig] int GetVolumeStepInfo(out uint step, out uint count);
        [PreserveSig] int ReservedStepUp(IntPtr context);
        [PreserveSig] int ReservedStepDown(IntPtr context);
        [PreserveSig] int QueryHardwareSupport(out uint mask);
        [PreserveSig] int GetVolumeRange(out float minimum, out float maximum, out float increment);
    }
    [ComImport, Guid("2A07407E-6497-4A18-9787-32F79BD0D98F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IDeviceTopology {
        [PreserveSig] int GetConnectorCount(out uint count);
        [PreserveSig] int GetConnector(uint index, out IConnector connector);
    }
    [ComImport, Guid("9C2C4058-23F5-41DE-877A-DF3AF236A09E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IConnector {
        [PreserveSig] int GetConnectorType(out int type);
        [PreserveSig] int GetDataFlow(out int flow);
        [PreserveSig] int ReservedConnectTo(IntPtr target);
        [PreserveSig] int ReservedDisconnect();
        [PreserveSig] int IsConnected([MarshalAs(UnmanagedType.Bool)] out bool connected);
        [PreserveSig] int GetConnectedTo(out IConnector connector);
    }
    [ComImport, Guid("AE2DE0E4-5BCA-4F2D-AA46-5D13F8FDB3A9"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IPart {
        [PreserveSig] int GetName([MarshalAs(UnmanagedType.LPWStr)] out string name);
        [PreserveSig] int GetLocalId(out uint id);
        [PreserveSig] int GetGlobalId([MarshalAs(UnmanagedType.LPWStr)] out string id);
        [PreserveSig] int GetPartType(out int type);
        [PreserveSig] int GetSubType(out Guid type);
        [PreserveSig] int GetControlInterfaceCount(out uint count);
        [PreserveSig] int GetControlInterface(uint index, out IntPtr result);
        [PreserveSig] int EnumPartsIncoming(out IPartsList parts);
        [PreserveSig] int EnumPartsOutgoing(out IPartsList parts);
        [PreserveSig] int GetTopologyObject(out IDeviceTopology topology);
        [PreserveSig] int Activate(uint context, ref Guid iid, [MarshalAs(UnmanagedType.IUnknown)] out object result);
    }
    [ComImport, Guid("6DAA848C-5EB0-45CC-AEA5-998A2CDA1FFB"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IPartsList {
        [PreserveSig] int GetCount(out uint count);
        [PreserveSig] int GetPart(uint index, out IPart part);
    }
    [ComImport, Guid("7FB7B48F-531D-44A2-BCB3-5AD5A134B3DC"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IVolumeLevel {
        [PreserveSig] int GetChannelCount(out uint count);
        [PreserveSig] int GetLevelRange(uint channel, out float minimum, out float maximum, out float increment);
        [PreserveSig] int GetLevel(uint channel, out float level);
    }

    public static class Probe {
        [DllImport("ole32.dll")] static extern int CoInitializeEx(IntPtr reserved, uint mode);
        [DllImport("ole32.dll")] static extern void CoUninitialize();
        [DllImport("ole32.dll")] static extern int PropVariantClear(ref PropVariant value);
        static void Check(int hr) { if (hr < 0) Marshal.ThrowExceptionForHR(hr); }
        static string Error(Exception ex) { return ex.GetType().Name + ": " + ex.Message + " [0x" + ex.HResult.ToString("X8") + "]"; }
        static void Release(object item) {
            if (item != null && Marshal.IsComObject(item)) {
                try { Marshal.ReleaseComObject(item); } catch (InvalidComObjectException) { }
            }
        }
        static object ReadProperty(IPropertyStore store, PropertyKey key) {
            PropVariant value = new PropVariant();
            try {
                Check(store.GetValue(ref key, out value));
                if (value.Type == 31) return Marshal.PtrToStringUni(value.Pointer);
                if (value.Type == 19) return value.UIntValue;
                if (value.Type == 0) return null;
                return "unsupported PROPVARIANT type " + value.Type;
            } finally { PropVariantClear(ref value); }
        }
        static object OptionalProperty(IPropertyStore store, PropertyKey key) {
            try { return ReadProperty(store, key); }
            catch (Exception ex) { return new { error = Error(ex) }; }
        }
        static object Volume(IDevice device) {
            object raw = null;
            try {
                Guid iid = typeof(IEndpointVolume).GUID;
                Check(device.Activate(ref iid, 23, IntPtr.Zero, out raw));
                IEndpointVolume volume = (IEndpointVolume)raw;
                bool mute; float scalar, db, minimum, maximum, increment; uint count, hardware;
                Check(volume.GetMute(out mute));
                Check(volume.GetMasterVolumeLevelScalar(out scalar));
                Check(volume.GetMasterVolumeLevel(out db));
                Check(volume.GetChannelCount(out count));
                Check(volume.GetVolumeRange(out minimum, out maximum, out increment));
                Check(volume.QueryHardwareSupport(out hardware));
                var channels = new List<object>();
                for (uint index = 0; index < count; index++) {
                    float channelDb, channelScalar;
                    Check(volume.GetChannelVolumeLevel(index, out channelDb));
                    Check(volume.GetChannelVolumeLevelScalar(index, out channelScalar));
                    channels.Add(new { channel = index, db = channelDb, scalar = channelScalar });
                }
                return new { mute = mute, scalar = scalar, db = db, min_db = minimum,
                    max_db = maximum, step_db = increment, hardware_support_mask = hardware,
                    channels = channels, note = "Slider/gain metadata; not input RMS or peak energy." };
            } catch (Exception ex) { return new { error = Error(ex) }; }
            finally { Release(raw); }
        }
        static void WalkIncoming(IPart part, HashSet<string> seen, List<object> controls, int depth) {
            if (depth > 32 || seen.Count >= 128) return;
            string id, name; uint localId; Guid subtype;
            Check(part.GetGlobalId(out id));
            if (!seen.Add(id)) return;
            Check(part.GetName(out name)); Check(part.GetLocalId(out localId)); Check(part.GetSubType(out subtype));
            object raw = null;
            try {
                Guid iid = typeof(IVolumeLevel).GUID;
                int hr = part.Activate(23, ref iid, out raw);
                if (hr >= 0 && raw != null) {
                    IVolumeLevel volume = (IVolumeLevel)raw;
                    uint count; Check(volume.GetChannelCount(out count));
                    var levels = new List<object>(); bool canAmplify = false;
                    for (uint channel = 0; channel < count; channel++) {
                        float db, minimum, maximum, increment;
                        Check(volume.GetLevel(channel, out db));
                        Check(volume.GetLevelRange(channel, out minimum, out maximum, out increment));
                        canAmplify = canAmplify || maximum > 0;
                        levels.Add(new { channel = channel, db = db, min_db = minimum, max_db = maximum, step_db = increment });
                    }
                    controls.Add(new { node_name = name, local_id = localId, subtype = subtype.ToString(),
                        can_amplify = canAmplify, levels = levels });
                }
            } finally { Release(raw); }
            IPartsList incoming = null;
            try {
                if (part.EnumPartsIncoming(out incoming) < 0 || incoming == null) return;
                uint count; Check(incoming.GetCount(out count));
                for (uint index = 0; index < count; index++) {
                    IPart child = null;
                    try { Check(incoming.GetPart(index, out child)); WalkIncoming(child, seen, controls, depth + 1); }
                    finally { Release(child); }
                }
            } finally { Release(incoming); }
        }
        static object Topology(IDevice device) {
            object raw = null; IConnector endpointConnector = null; IConnector adapterConnector = null;
            var controls = new List<object>(); var seen = new HashSet<string>();
            try {
                Guid iid = typeof(IDeviceTopology).GUID;
                Check(device.Activate(ref iid, 23, IntPtr.Zero, out raw));
                IDeviceTopology topology = (IDeviceTopology)raw;
                uint count; Check(topology.GetConnectorCount(out count));
                if (count == 0) return new { status = "no connector", volume_controls = controls };
                Check(topology.GetConnector(0, out endpointConnector));
                Check(endpointConnector.GetConnectedTo(out adapterConnector));
                WalkIncoming((IPart)adapterConnector, seen, controls, 0);
                return new { status = "queried", visited_parts = seen.Count, volume_controls = controls,
                    note = "Upstream hardware controls; positive max_db indicates gain capability. No control exposed does not prove boost is off. Mux branch selection is not inferred." };
            } catch (Exception ex) { return new { status = "unavailable_or_partial", error = Error(ex), volume_controls = controls }; }
            finally { Release(adapterConnector); Release(endpointConnector); Release(raw); }
        }
        public static object ReadCaptureEndpoints(bool includeTopology) {
            int initialized = CoInitializeEx(IntPtr.Zero, 2);
            bool uninitialize = initialized >= 0;
            if (initialized < 0 && initialized != unchecked((int)0x80010106)) Check(initialized);
            IDeviceEnumerator enumerator = null; IDeviceCollection collection = null;
            try {
                enumerator = (IDeviceEnumerator)Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")));
                var defaults = new Dictionary<string, object>();
                var defaultIds = new Dictionary<int, string>();
                string[] roles = new string[] { "console", "multimedia", "communications" };
                for (int role = 0; role < roles.Length; role++) {
                    IDevice device = null;
                    try {
                        Check(enumerator.GetDefaultAudioEndpoint(1, role, out device));
                        string id; Check(device.GetId(out id)); defaults[roles[role]] = id; defaultIds[role] = id;
                    } catch (Exception ex) { defaults[roles[role]] = new { error = Error(ex) }; }
                    finally { Release(device); }
                }
                Check(enumerator.EnumAudioEndpoints(1, 1, out collection));
                uint count; Check(collection.GetCount(out count));
                var endpoints = new List<object>();
                for (uint index = 0; index < count; index++) {
                    IDevice device = null; IPropertyStore store = null;
                    var row = new Dictionary<string, object>();
                    try {
                        Check(collection.Item(index, out device));
                        string id; uint state; Check(device.GetId(out id)); Check(device.GetState(out state));
                        row["endpoint_id"] = id; row["state"] = state;
                        var matches = new List<string>();
                        foreach (var pair in defaultIds) if (pair.Value == id) matches.Add(roles[pair.Key]);
                        row["default_roles"] = matches;
                        row["volume"] = Volume(device);
                        if (device.OpenPropertyStore(0, out store) >= 0) { // STGM_READ only.
                            row["name"] = OptionalProperty(store, new PropertyKey("A45C254E-DF1C-4EFD-8020-67D146A850E0", 14));
                            row["form_factor"] = OptionalProperty(store, new PropertyKey("1DA5D803-D492-4EDD-8C23-E0C0FFEE7F0E", 0));
                            row["sysfx_disable_property"] = OptionalProperty(store, new PropertyKey("1DA5D803-D492-4EDD-8C23-E0C0FFEE7F0E", 5));
                        } else { row["property_store"] = "unavailable"; }
                        row["sysfx_interpretation"] = "0=enabled, 1=disabled, null=not exposed; does not enumerate every vendor enhancement.";
                        row["hardware_topology"] = includeTopology ? Topology(device) : (object)"skipped";
                    } catch (Exception ex) { row["error"] = Error(ex); }
                    finally { Release(store); Release(device); }
                    endpoints.Add(row);
                }
                return new { readonly_query = true, audio_capture = false, audio_playback = false,
                    settings_changed = false, defaults = defaults, capture_endpoints = endpoints,
                    listen_to_device = "unknown: no supported general getter used; no registry or private audio-policy interfaces accessed." };
            } finally { Release(collection); Release(enumerator); if (uninitialize) CoUninitialize(); }
        }
    }
}
'@
}

if ($CompileOnly) {
    [pscustomobject]@{ compiled = $true; endpoint_queries_executed = $false } | ConvertTo-Json
    return
}
[FanyiJiEndpointReadOnly.Probe]::ReadCaptureEndpoints(-not $SkipTopology.IsPresent) | ConvertTo-Json -Depth 14
