# INCOIS certificate chain

INCOIS currently sends its leaf certificate without its intermediate. The
archiver loads `globalsign-rsa-ov-ssl-ca-2018.pem` alongside the default system
trust store so certificate and hostname verification remain enabled.

Source: https://secure.globalsign.com/cacert/gsrsaovsslca2018.crt
(the CA Issuers URL in the INCOIS leaf certificate, fetched over HTTPS).

Subject: GlobalSign RSA OV SSL CA 2018

Issuer: GlobalSign Root CA - R3

Valid through: 2028-11-21

SHA-256 certificate fingerprint:
`B676FFA3179E8812093A1B5EAFEE876AE7A6AAF231078DAD1BFB21CD2893764A`

If INCOIS changes issuer or this intermediate expires, obtain the replacement
from the issuing CA over verified HTTPS and update this file and the PEM.
