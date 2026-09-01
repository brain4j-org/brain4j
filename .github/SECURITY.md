# Security Policy for brain4j

## Reporting a Vulnerability

The brain4j team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings.

If you believe you've found a security vulnerability in brain4j, please follow these steps:

1. **Do not disclose the vulnerability publicly**
2. **Contact us directly** at https://t.me/xecho1337 with details about the vulnerability
3. Include the following information in your report:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

## Response Timeline

- **Initial Response**: We aim to acknowledge receipt of vulnerability reports within 72-96 hours
- **Status Update**: We will provide an update on the vulnerability within 14 days
- **Vulnerability Fix**: The timeline for fixing the vulnerability will depend on its severity and complexity

## Supported Versions
| Version | Supported          |
|---------|--------------------|
| ≥ 2.5.3 | :white_check_mark: |
| < 2.5.3 | :x:                |

## Security Best Practices for Using brain4j

### Data Protection
- Models trained with brain4j may contain sensitive information from training data
- Always sanitize and anonymize sensitive data before using it for training
- Consider data minimization principles when developing ML applications

### Model Security
- Be cautious when using models from untrusted sources
- Validate inputs to prevent injection attacks or model manipulation
- Consider implementing rate limiting for prediction endpoints

### Native Code Considerations
Since brain4j contains native C (OpenCL) code, users should be aware of:
- Potential memory safety concerns when using native functionality
- The importance of keeping the library updated to receive security patches
- Additional system-level security implications

## Security Development Process

The brain4j team follows these practices to minimize security risks:

1. Regular code reviews with security focus
2. Automated testing and static analysis
3. Dependency vulnerability scanning
4. Periodic security assessments

----

For non-security related issues, please use the [GitHub Issues](https://github.com/xEcho1337/brain4j/issues) page.
<br>
Last updated: 2025-03-30
