/**
 *  Copyright (c) 2015 by Contributors
 * @file   network_utils.h
 * @brief  network utilities
 */
#ifndef PS_NETWORK_UTILS_H_
#define PS_NETWORK_UTILS_H_
#include <unistd.h>
#ifdef _MSC_VER
#include <tchar.h>
#include <winsock2.h>
#include <windows.h>
#include <iphlpapi.h>
#undef interface
#else
#include <net/if.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#endif
#include <string>

namespace ps {

/**
 * \brief return the IP address for given interface eth0, eth1, ...
 */
void GetIP(const std::string& interface, std::string* ip) {
#ifdef _MSC_VER
  typedef std::basic_string<TCHAR> tstring;
  // Try to get the Adapters-info table, so we can given useful names to the IP
  // addresses we are returning.  Gotta call GetAdaptersInfo() up to 5 times to handle
  // the potential race condition between the size-query call and the get-data call.
  // I love a well-designed API :^P
  IP_ADAPTER_INFO * pAdapterInfo = NULL;
  {
    ULONG bufLen = 0;
    for (int i = 0; i < 5; i++) {
      DWORD apRet = GetAdaptersInfo(pAdapterInfo, &bufLen);
      if (apRet == ERROR_BUFFER_OVERFLOW) {
        free(pAdapterInfo);  // in case we had previously allocated it
        pAdapterInfo = static_cast<IP_ADAPTER_INFO*>(malloc(bufLen));
      } else if (apRet == ERROR_SUCCESS) {
        break;
      } else {
        free(pAdapterInfo);
        pAdapterInfo = NULL;
        break;
      }
    }
  }
  if (pAdapterInfo) {
    tstring keybase = _T(
        "SYSTEM\\CurrentControlSet\\Control\\Network\\{4D36E972-E325-11CE-BFC1-08002BE10318}\\");
    tstring connection = _T("\\Connection");

    IP_ADAPTER_INFO *curpAdapterInfo = pAdapterInfo;
    while (curpAdapterInfo) {
      HKEY hKEY;
      std::string AdapterName = curpAdapterInfo->AdapterName;
      // GUID only ascii
      tstring key_set = keybase + tstring(AdapterName.begin(), AdapterName.end()) + connection;
      LPCTSTR data_Set = key_set.c_str();
      LPCTSTR dwValue = NULL;
      if (ERROR_SUCCESS ==
          ::RegOpenKeyEx(HKEY_LOCAL_MACHINE, data_Set, 0, KEY_READ, &hKEY)) {
        DWORD dwSize = 0;
        DWORD dwType = REG_SZ;
        if (ERROR_SUCCESS ==
            ::RegQueryValueEx(hKEY, _T("Name"), 0, &dwType, (LPBYTE)dwValue, &dwSize)) {
          dwValue = new TCHAR[dwSize];
          if (ERROR_SUCCESS ==
              ::RegQueryValueEx(hKEY, _T("Name"), 0, &dwType, (LPBYTE)dwValue, &dwSize)) {
            // interface name must only ascii
            tstring tstr = dwValue;
            std::string s(tstr.begin(), tstr.end());
            if (s == interface) {
              *ip = curpAdapterInfo->IpAddressList.IpAddress.String;
              break;
            }
          }
        }
        ::RegCloseKey(hKEY);
      }
      curpAdapterInfo = curpAdapterInfo->Next;
    }
    free(pAdapterInfo);
  }
#else
  struct ifaddrs * ifAddrStruct = NULL;
  struct ifaddrs * ifa = NULL;
  void * tmpAddrPtr = NULL;

  getifaddrs(&ifAddrStruct);
  for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == NULL) continue;
    if (ifa->ifa_addr->sa_family == AF_INET) {
      // is a valid IP4 Address
      tmpAddrPtr = &(reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr))->sin_addr;
      char addressBuffer[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
      if (strncmp(ifa->ifa_name,
                  interface.c_str(),
                  interface.size()) == 0) {
        *ip = addressBuffer;
        break;
      }
    }
  }
  if (ifAddrStruct != NULL) freeifaddrs(ifAddrStruct);
#endif
}


/**
 * \brief return the IP address and Interface the first interface which is not
 * loopback
 *
 * only support IPv4
 */
void GetAvailableInterfaceAndIP(
    std::string* interface, std::string* ip) {
#ifdef _MSC_VER
  typedef std::basic_string<TCHAR> tstring;
  IP_ADAPTER_INFO * pAdapterInfo = NULL;
  {
    ULONG bufLen = 0;
    for (int i = 0; i < 5; i++) {
      DWORD apRet = GetAdaptersInfo(pAdapterInfo, &bufLen);
      if (apRet == ERROR_BUFFER_OVERFLOW) {
        free(pAdapterInfo);  // in case we had previously allocated it
        pAdapterInfo = static_cast<IP_ADAPTER_INFO*>(malloc(bufLen));
      } else if (apRet == ERROR_SUCCESS) {
        break;
      } else {
        free(pAdapterInfo);
        pAdapterInfo = NULL;
        break;
      }
    }
  }
  if (pAdapterInfo) {
    tstring keybase = _T(
        "SYSTEM\\CurrentControlSet\\Control\\Network\\{4D36E972-E325-11CE-BFC1-08002BE10318}\\");
    tstring connection = _T("\\Connection");

    IP_ADAPTER_INFO *curpAdapterInfo = pAdapterInfo;
    HKEY hKEY = NULL;
    while (curpAdapterInfo) {
      std::string curip(curpAdapterInfo->IpAddressList.IpAddress.String);
      if (curip == "127.0.0.1") {
        curpAdapterInfo = curpAdapterInfo->Next;
        continue;
      }
      if (curip == "0.0.0.0") {
        curpAdapterInfo = curpAdapterInfo->Next;
        continue;
      }

      std::string AdapterName = curpAdapterInfo->AdapterName;
      // GUID only ascii
      tstring key_set = keybase + tstring(AdapterName.begin(), AdapterName.end()) + connection;
      LPCTSTR data_Set = key_set.c_str();
      LPCTSTR dwValue = NULL;
      if (ERROR_SUCCESS ==
          ::RegOpenKeyEx(HKEY_LOCAL_MACHINE, data_Set, 0, KEY_READ, &hKEY)) {
        DWORD dwSize = 0;
        DWORD dwType = REG_SZ;
        if (ERROR_SUCCESS ==
            ::RegQueryValueEx(hKEY, _T("Name"), 0, &dwType, (LPBYTE)dwValue, &dwSize)) {
          dwValue = new TCHAR[dwSize];
          if (ERROR_SUCCESS ==
              ::RegQueryValueEx(hKEY, _T("Name"), 0, &dwType, (LPBYTE)dwValue, &dwSize)) {
            // interface name must only ascii
            tstring tstr = dwValue;
            std::string s(tstr.begin(), tstr.end());

            *interface = s;
            *ip = curip;
            break;
          }
        }
        ::RegCloseKey(hKEY);
        hKEY = NULL;
      }
      curpAdapterInfo = curpAdapterInfo->Next;
    }
    if (hKEY != NULL) {
      ::RegCloseKey(hKEY);
    }
    free(pAdapterInfo);
  }
#else
  struct ifaddrs * ifAddrStruct = nullptr;
  struct ifaddrs * ifa = nullptr;

  interface->clear();
  ip->clear();
  getifaddrs(&ifAddrStruct);
  for (ifa = ifAddrStruct; ifa != nullptr; ifa = ifa->ifa_next) {
    if (nullptr == ifa->ifa_addr) continue;

    if (AF_INET == ifa->ifa_addr->sa_family &&
        0 == (ifa->ifa_flags & IFF_LOOPBACK)) {
      char address_buffer[INET_ADDRSTRLEN];
      void* sin_addr_ptr = &(reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr))->sin_addr;
      inet_ntop(AF_INET, sin_addr_ptr, address_buffer, INET_ADDRSTRLEN);

      *ip = address_buffer;
      *interface = ifa->ifa_name;

      break;
    }
  }
  if (nullptr != ifAddrStruct) freeifaddrs(ifAddrStruct);
  return;
#endif
}

/**
 * \brief return an available port on local machine
 *
 * only support IPv4
 * \return 0 on failure
 */
int GetAvailablePort() {
  struct sockaddr_in addr;
  addr.sin_port = htons(0);  // have system pick up a random port available for me
  addr.sin_family = AF_INET;  // IPV4
  addr.sin_addr.s_addr = htonl(INADDR_ANY);  // set our addr to any interface

  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (0 != bind(sock, (struct sockaddr*)&addr, sizeof(struct sockaddr_in))) {
    perror("bind():");
    return 0;
  }
#ifdef _MSC_VER
  int addr_len = sizeof(struct sockaddr_in);
#else
  socklen_t addr_len = sizeof(struct sockaddr_in);
#endif

  if (0 != getsockname(sock, (struct sockaddr*)&addr, &addr_len)) {
    perror("getsockname():");
    return 0;
  }

  int ret_port = ntohs(addr.sin_port);
#ifdef  _MSC_VER
  closesocket(sock);
#else
  close(sock);
#endif
  return ret_port;
}

}  // namespace ps
#endif  // PS_NETWORK_UTILS_H_
