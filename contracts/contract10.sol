
pragma solidity ^0.8.12;

interface IERC777 {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function totalSupply() external view returns (uint256);
    function balanceOf(address owner) external view returns (uint256);
    function send(address recipient, uint256 amount, bytes calldata data) external;
    function transfer(address recipient, uint256 amount) external returns (bool);
    function authorizeOperator(address operator) external;
    function revokeOperator(address operator) external;
    function isOperatorFor(address operator, address tokenHolder) external view returns (bool);
    function operatorSend(address sender, address recipient, uint256 amount, bytes calldata data, bytes calldata operatorData) external;
    function mint(address recipient, uint256 amount, bytes calldata data) external;
    function burn(uint256 amount, bytes calldata data) external;

    event Sent(address indexed operator, address indexed from, address indexed to, uint256 amount, bytes data, bytes operatorData);
    event Minted(address indexed operator, address indexed to, uint256 amount, bytes data, bytes operatorData);
    event Burned(address indexed operator, address indexed from, uint256 amount, bytes data, bytes operatorData);
    event AuthorizedOperator(address indexed operator, address indexed tokenHolder);
    event RevokedOperator(address indexed operator, address indexed tokenHolder);
}

interface IERC20 {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

interface IERC1820Registry {
    function setInterfaceImplementer(address account, bytes32 interfaceHash, address implementer) external;
    function getInterfaceImplementer(address account, bytes32 interfaceHash) external view returns (address);
    function setManager(address account, address newManager) external;
    function getManager(address account) external view returns (address);
}

contract Ownable {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(_owner == msg.sender, "Ownable: caller is not the owner");
        _;
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

contract RedRubyClubTokenContract is IERC777, IERC20, Ownable {
    string private _name;
    string private _symbol;
    uint256 private _totalSupply;
    uint8 private _decimals;
    address public burnRecipient;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => bool)) private _authorizedOperators;
    mapping(address => mapping(address => uint256)) private _allowances;

    IERC1820Registry private constant _ERC1820_REGISTRY = IERC1820Registry(0x1820a4B7618BdE71Dce8cdc73aAB6C95905faD24);

    bytes32 private constant _TOKENS_SENDER_INTERFACE_HASH = keccak256("ERC777TokensSender");
    bytes32 private constant _TOKENS_RECIPIENT_INTERFACE_HASH = keccak256("ERC777TokensRecipient");


    constructor(string memory name_, string memory symbol_, uint256 initialSupply, address[] memory defaultOperators) {
        _name = name_;
        _symbol = symbol_;
        _decimals = 18;
        _mint(msg.sender, initialSupply, "", "");

        bool success;
        bytes memory data;

        (success, data) = address(_ERC1820_REGISTRY).call(
            abi.encodeWithSignature("setInterfaceImplementer(address,bytes32,address)", address(this), keccak256("ERC777Token"), address(this))
        );
        require(success, "Failed to set ERC777Token interface implementer");

        (success, data) = address(_ERC1820_REGISTRY).call(
            abi.encodeWithSignature("setInterfaceImplementer(address,bytes32,address)", address(this), keccak256("ERC20Token"), address(this))
        );
        require(success, "Failed to set ERC20Token interface implementer");

        for (uint256 i = 0; i < defaultOperators.length; i++) {
            _authorizedOperators[defaultOperators[i]][msg.sender] = true;
        }
    }

    function name() public view override(IERC777, IERC20) returns (string memory) {
        return _name;
    }

    function symbol() public view override(IERC777, IERC20) returns (string memory) {
        return _symbol;
    }

    function decimals() public view override returns (uint8) {
        return _decimals;
    }

    function totalSupply() public view override(IERC777, IERC20) returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address owner) public view override(IERC777, IERC20) returns (uint256) {
        return _balances[owner];
    }

    function send(address recipient, uint256 amount, bytes calldata data) public override {
        _send(msg.sender, msg.sender, recipient, amount, data, "", true);
    }

    function transfer(address recipient, uint256 amount) public override(IERC777, IERC20) returns (bool) {
        _send(msg.sender, msg.sender, recipient, amount, "", "", false);
        return true;
    }

    function authorizeOperator(address operator) public override {
        require(msg.sender != operator, "ERC777: authorizing self as operator");
        _authorizedOperators[operator][msg.sender] = true;
        emit AuthorizedOperator(operator, msg.sender);
    }

    function revokeOperator(address operator) public override {
        require(operator != msg.sender, "ERC777: revoking self as operator");
        _authorizedOperators[operator][msg.sender] = false;
        emit RevokedOperator(operator, msg.sender);
    }

    function isOperatorFor(address operator, address tokenHolder) public view override returns (bool) {
        return operator == tokenHolder || _authorizedOperators[operator][tokenHolder];
    }

    function operatorSend(
        address sender,
        address recipient,
        uint256 amount,
        bytes calldata data,
        bytes calldata operatorData
    ) public override {
        require(isOperatorFor(msg.sender, sender), "ERC777: caller is not an operator for holder");
        _send(msg.sender, sender, recipient, amount, data, operatorData, true);
    }

    function mint(address recipient, uint256 amount, bytes calldata data) public override onlyOwner {
        _mint(recipient, amount, data, "");
    }

    function setBurnRecipient(address _burnRecipient) external onlyOwner {
        burnRecipient = _burnRecipient;
    }

    function burn(uint256 amount, bytes calldata data) public override onlyOwner {
        _burn(msg.sender, amount, data, "");
    }

    function burntoAddress(uint256 amount, bytes calldata data) public onlyOwner {
        require(burnRecipient != address(0), "Burn recipient is not set");
        _send(msg.sender, msg.sender, burnRecipient, amount, data, "", false);
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        require(_allowances[sender][msg.sender] >= amount, "ERC777: transfer amount exceeds allowance");
        _allowances[sender][msg.sender] -= amount;
        _send(msg.sender, sender, recipient, amount, "", "", false);
        return true;
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function _send(
        address operator,
        address from,
        address to,
        uint256 amount,
        bytes memory data,
        bytes memory operatorData,
        bool requireReceptionAck
    ) private {
        require(from != address(0), "ERC777: send from the zero address");
        require(to != address(0), "ERC777: send to the zero address");

        _callTokensToSend(operator, from, to, amount, data, operatorData);

        _move(operator, from, to, amount, data, operatorData);

        _callTokensReceived(operator, from, to, amount, data, operatorData, requireReceptionAck);
    }

    function _move(
        address operator,
        address from,
        address to,
        uint256 amount,
        bytes memory data,
        bytes memory operatorData
    ) private {
        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "ERC777: transfer amount exceeds balance");
        _balances[from] = fromBalance - amount;
        _balances[to] += amount;

        emit Sent(operator, from, to, amount, data, operatorData);
        emit Transfer(from, to, amount); // ERC20 compatibility
    }

    function _mint(
        address account,
        uint256 amount,
        bytes memory userData,
        bytes memory operatorData
    ) internal {
        require(account != address(0), "ERC777: mint to the zero address");

        _totalSupply += amount;
        _balances[account] += amount;

        emit Minted(msg.sender, account, amount, userData, operatorData);
        emit Transfer(address(0), account, amount); // ERC20 compatibility
    }

    function _burn(
        address from,
        uint256 amount,
        bytes memory data,
        bytes memory operatorData
    ) internal {
        require(from != address(0), "ERC777: burn from the zero address");

        _callTokensToSend(msg.sender, from, address(0), amount, data, operatorData);

        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "ERC777: burn amount exceeds balance");
        _balances[from] = fromBalance - amount;
        _totalSupply -= amount;

        emit Burned(msg.sender, from, amount, data, operatorData);
        emit Transfer(from, address(0), amount); // ERC20 compatibility
    }

    function _callTokensToSend(
        address operator,
        address from,
        address to,
        uint256 amount,
        bytes memory data,
        bytes memory operatorData
    ) private {
        bool success;
        bytes memory returnData;

        (success, returnData) = address(_ERC1820_REGISTRY).staticcall(
            abi.encodeWithSignature("getInterfaceImplementer(address,bytes32)", from, _TOKENS_SENDER_INTERFACE_HASH)
        );
        if (success && returnData.length > 0) {
            address implementer = abi.decode(returnData, (address));
            if (implementer != address(0)) {
                IERC777Sender(implementer).tokensToSend(operator, from, to, amount, data, operatorData);
            }
        }
    }

    function _callTokensReceived(
        address operator,
        address from,
        address to,
        uint256 amount,
        bytes memory data,
        bytes memory operatorData,
        bool requireReceptionAck
    ) private {
        bool success;
        bytes memory returnData;

        (success, returnData) = address(_ERC1820_REGISTRY).staticcall(
            abi.encodeWithSignature("getInterfaceImplementer(address,bytes32)", to, _TOKENS_RECIPIENT_INTERFACE_HASH)
        );
        if (success && returnData.length > 0) {
            address implementer = abi.decode(returnData, (address));
            if (implementer != address(0)) {
                IERC777Recipient(implementer).tokensReceived(operator, from, to, amount, data, operatorData);
            } else if (requireReceptionAck) {
                require(to.code.length == 0, "ERC777: token recipient contract has no implementer for ERC777TokensRecipient");
            }
        } else if (requireReceptionAck) {
            require(to.code.length == 0, "ERC777: token recipient contract has no implementer for ERC777TokensRecipient");
        }
    }
}

interface IERC777Sender {
    function tokensToSend(address operator, address from, address to, uint256 amount, bytes calldata userData, bytes calldata operatorData) external;
}

interface IERC777Recipient {
    function tokensReceived(address operator, address from, address to, uint256 amount, bytes calldata userData, bytes calldata operatorData) external;
}